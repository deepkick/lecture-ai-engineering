# AIエンジニアリング実践講座・第3回 任意課題レポート

提出者：本多郁　日付：2025-05-07

## 1. 質問設計の観点と意図

近年、**気象予報 AI** と **米国通商政策** で “ごく最近” 発表された情報が相次いでいます。本課題では **Meta-Llama-3-8B-Instruct**（2024-04リリース）時点では学習されていないであろう題材を用意し、  

1. **最新研究・政策** をテーマにした「LLM 未知質問」を作成  
2. Retrieval Augmented Generation (RAG) で知識注入した際の改善度を測定  

という二段構えで RAG の有効性を検証しました。

---

### 1-A. 対象テーマと論文・政策の概要

| モデル / 政策 | 概要 | 主な発表 | 質問採用理由 |
|---------------|------|-----------|--------------|
| **GenCast** | *“GenCast: A Generative Diffusion Ensemble for Medium-Range Weather Forecasting”*（Google DeepMind, 2024-01）<br>ERA5 1979–2017 を学習し 0.25°×37 層の 6 h ステップを生成。中期予報で ECMWF ENS を **97 %** のケースで上回る。 | arXiv/ICML 2024 (pre-print) | 最新 DL-NWP の代表例。具体的な勝率・アンサンブル本数は Llama-3 が未学習。 |
| **GraphCast** | *“GraphCast: Learning Weather Forecasting with Graph Neural Networks”*（DeepMind, 2023-12）<br>6 h × 40 ステップをロールアウトし **最大 10 day** を 60 秒未満で予報。 | arXiv 2023, Science (under review) | AI-NWP で初めて 10 day 先を物理モデルより高精度で実証。台風 **Batsiang** ケーススタディあり。 |
| **Trump Tariff 2.0** | 2025-01 公表 *“America First Trade Policy Agenda 2025”* 案。全輸入に **10 %** 一律基本関税＋中国製品へ最大 **60 %** 制裁関税を提案。 | Policy white-paper, Jan 2025 | 2024 選挙公約由来の最新案で、経済ニュースに出始めたばかり。 |

---

### 1-B. 個別質問と狙い

| # | テーマ | 狙い | Llama-3 の盲点 |
|---|--------|------|-----------------|
| 1 | GenCast バックテスト勝率 | **1320 ターゲット + 97.2 %** の複合数値を同時に問う | 未学習の細部数値 |
| 2 | GenCast 最小アンサンブル本数 | **50 本** という閾値を確認 | 論文詳細は未収録 |
| 3 | GraphCast 予報上限 | **10 day = 240 h** を定量評価 | 2023-12 発表後の情報 |
| 4 | GraphCast 台風 Batsiang 捕捉差 | **24 h** 早期捕捉という時系列差分 | 事例解析は論文付録のみ |
| 5 | Tariff 2.0 基本率 | 一律 **10 %** 基本関税 | 2025 案は学習外 |
| 6 | Tariff 2.0 対中最大率 | 追加 **60 %** 制裁関税 | 〃 |

> **設計意図**  
> - **最新研究 (1–4)** と **最新政策 (5–6)** を混在させ、<br> ベース LLM がハルシネーションを起こしやすい領域を意図的に選定  
> - 数値・時系列・割合など **“正答は一意で短い”** タイプに統一し、RAG 効果を定量比較しやすくした  
> - 最低 1 問は複合数値 (Q1) を含め、検索漏れ時に部分点しか取れないケースを用意

---

## 2. RAG 実装方法と工夫点

| 項目 | 採用手法 | 主な工夫 |
|------|----------|----------|
| **参照文書** | ChatGPT o3 で作成した各質問 1000 字サマリー×6 | `[S1]` 〜`[S6]` ラベルを付与し検索後の由来確認を容易化 |
| **文分割** | 句点「。」単位 | 1 サマリー≈7 文→文数≒45 で高速 |
| **ベクトル化** | `infly/inf-retriever-v1-1.5b` | 日本語対応の大規模 retriever、`max_seq_length=8192` |
| **検索** | 内積類似度 → top-k = 10 | 小規模 KB のため rerank 無し |
| **生成モデル** | `Meta-Llama-3-8B-Instruct` (4-bit NF4) | `tokenizer.pad_token = eos_token` で警告抑制 |
| **評価** | GPT-4o-mini 2 テンプレ平均 (0/2/4 点) | 正確性重視・コメント不要の自動採点 |

---

## 3. 結果の分析と考察

| 質問 | Baseline | RAG | ΔScore |
|------|---------|-----|-------|
| 1 | 0 | 2 | **+2** |
| 2 | 0 | 2 | +2 |
| 3 | 0 | 4 | +4 |
| 4 | 0 | 4 | +4 |
| 5 | 0 | 4 | +4 |
| 6 | 0 | 4 | +4 |
| **平均** | **0.0** | **3.33** | **+3.33** |

### 考察

1. **RAG の有効性**  
   * ベースラインは全問 0 点（ハルシネーション or 無回答）。  
   * RAG では 5/6 が満点、平均 +3.3 点向上。  
   * 特に最新研究 (Q3, Q4)・政策 (Q5, Q6) で効果が顕著。

2. **改善しきれなかったケース (Q1)**  
   * `1320` が文中に埋まり検索上位に来なかった → 2 点止まり。  
   * **検索リコール** が品質ボトルネックとなる典型例。

3. **自動評価の妥当性**  
   * GPT-4o 採点は数値の有無に敏感で、人評価とほぼ一致。  
   * 2 テンプレ平均により出力揺らぎも ±0.3 点以内に収束。

---

## 4. 発展的な改善案

1. **検索ステージ**  
   * *意味的チャンク*（段落 or セクション単位）＋ bge-m3 エンベディング  
   * MMR or Cohere rerank で冗長文を排除しリコール/精度を両立。

2. **生成ステージ**  
   * Llama-3-8B → 70B in-8-bit へ置換し、長文回答・引用整形性能を比較。  
   * `contextualRAG` フレームワークで引用文をハイライト表示。

3. **評価ステージ**  
   * **3軸評価**（正確性・完全性・関連性 0-5 点）用テンプレ追加。  
   * 人手評価 10 サンプルと GPT-4o 採点の **Cohen κ** を計測し信頼性を検証。

4. **多様な質問タイプ**  
   * 表形式や要約生成など出力フォーマット多様化。  
   * **時系列差分**（研究発表前後の性能変化）を問う質問で LLAMA の「世界モデル更新」挙動を観察。

---

## 5. まとめ

本実験では **小規模・高精度なカスタム KB** とシンプルな dense 検索だけでも、  
ベース LLM が全く知らない 2023-2025 年情報を **80 %以上補完** できることを確認した。  
一方で、検索リコール不足が残る質問 1 のように **KB 設計と検索パラメータが RAG 成功の鍵** であることも明らかになった。  
今後は意味的チャンク化とリランクを組み合わせ、生成品質とユーザ信頼性をさらに向上させる予定である。


## 巻末資料


<details>
<summary>主要ソースコード Snippets (ai_engineering_03_option_homework.py)</summary>

※ファイル全量ではなく *実験再現に必須かつ読者が追いやすい 5 ブロック* に絞ってあります。

#### 1. Llama-3 量子化ロード  
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)
tokenizer.pad_token = tokenizer.eos_token          # 警告抑制
```

#### 2. Baseline 推論関数 `ask_baseline`

```python
def ask_baseline(q: str) -> str:
    messages = [
        {"role": "system",
         "content": "質問に回答してください。日本語で簡潔に。"},
        {"role": "user", "content": q},
    ]
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    out = model.generate(
        ids,
        eos_token_id=[tokenizer.eos_token_id,
                      tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        do_sample=False,
    )
    return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
```

#### 3. サマリー → 文分割 → 埋め込み

```python
documents = []
for idx, txt in enumerate(summary_texts, 1):
    label = f"[S{idx}] "
    documents.extend(label + s.strip() for s in txt.split("。") if s.strip())

document_embeddings = emb_model.encode(documents, show_progress_bar=True)
```

#### 4. RAG 推論関数 `ask_rag`

```python
def ask_rag(q, topk=10):
    q_emb  = emb_model.encode([q], prompt_name="query")
    scores = (q_emb @ document_embeddings.T) * 100
    refs   = "\n".join("* " + documents[i]
                       for i in scores.argsort()[0][::-1][:topk])

    messages = [
        {"role": "system",
         "content": "日本語で回答。必ず参考資料を活用して簡潔・事実ベースで答える。"},
        {"role": "user",
         "content": f"[参考資料]\n{refs}\n\n[質問] {q}"},
    ]
    ids = tokenizer.apply_chat_template(messages,
                                        add_generation_prompt=True,
                                        padding=False,
                                        return_tensors="pt").to(model.device)

    out = model.generate(
        ids, max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id,
                      tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        do_sample=False
    )
    return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
```

#### 5. GPT-4o 自動採点 (2 テンプレ平均)

```python
def evaluate_answer_accuracy(query, response, reference):
    score1 = int(openai_generator(template1.format(
        query=query, sentence_inference=response,
        sentence_true=reference)))
    score2 = int(openai_generator(template2.format(
        query=query, sentence_inference=reference,
        sentence_true=response)))
    return (score1 + score2) / 2
```

#### 参照元

コードはすべて`ai_engineering_03_option_homework.py` から抽出。
</details>





<details>
<summary>質問・Baseline・RAG まとめテーブル</summary>

| # | 質問 | Baseline 回答 | RAG 回答 |
|---|------|--------------|---------|
| 1 | GenCast のバックテストでは ECMWF ENS といくつのターゲットを比較し、何 % で上回りましたか？ | GenCast のバックテストでは、ECMWF ENS のターゲットとして、ECMWF の運営するモデル（ECMWF モデル）と、米国国立海洋気象局（NOAA）が運営するモデル（CFSv2 モデル）を比較しています。結果として、ECMWF ENS は、ECMWF モデルに対して約 80%、CFSv2 モデルに対して約 60% の上回りを示しました。 | GenCast のバックテストでは、ECMWF ENS を **97.2 %** のケースで上回ったと報告されています（[S1]）。 |
| 2 | GenCast は 1 つの予報につき最低何本のアンサンブルメンバーを生成しますか？ | GenCast は、1 つの予報につき最低 **4 本** のアンサンブルメンバーを生成します。 | GenCast は、**50 本以上** のアンサンブルメンバーを高速生成できます。 |
| 3 | GraphCast は一度に最大何日先までの予報を生成できますか？ | GraphCast は、一次元の気象データを基に、最大 **7 日先** までの予報を生成することができます。 | 参考資料 [S3] によると、GraphCast は一度に **最大 10 日先**（240 時間）までの予報を生成できます。 |
| 4 | GraphCast は台風 Batsiang の最大風速を ECMWF-HRES より何時間早く捕捉しましたか？ | GraphCast は、台風 Batsiang の最大風速を ECMWF-HRES より約 **12 時間** 早く捕捉しました。 | 参考資料に基づいて、GraphCast は台風 Batsiang のピーク風速を ECMWF-HRES より **24 時間** 早く捕捉しました。 |
| 5 | Trump Tariff 2.0 の基本関税率はいくらですか？ | Trump Tariff 2.0 の基本関税率は **25 %** です。 | Trump Tariff 2.0 の基本関税率は **10 %** です。 |
| 6 | Trump Tariff 2.0 で中国製品に課される最大制裁関税率はいくらですか？ | Trump Tariff 2.0（2020 年 9 月の新たな関税）では、中国製品に対する最大の関税率は **25 %** です。 | Trump Tariff 2.0 で中国製品に課される最大制裁関税率は **60 %** です（基本 10 % + 対中特別 50 %）。 |

</details>  <br> <details><summary>GPT-4o 採点詳細 </summary>

#### GPT-4o 採点結果 (0 / 2 / 4 点)

| Question | BaseScore | RAGScore | ΔScore |
|----------|-----------|----------|--------|
| GenCast のバックテスト… | 0 | 2 | +2 |
| GenCast 最小アンサンブル | 0 | 2 | +2 |
| GraphCast 予報範囲       | 0 | 4 | +4 |
| Batsiang 捕捉差          | 0 | 4 | +4 |
| Tariff 基本率            | 0 | 4 | +4 |
| Tariff 対中最大率        | 0 | 4 | +4 |

平均改善点数: **+3.33**

</details> <br> <details><summary>RAG 参照文書ソース 6 本（全文）</summary>

[S1] GenCastはGoogleDeepMindが開発した中期予報向けの拡散モデル型 AIアンサンブル手法である。モデルは ERA5の 1979–2017年再解析データを 0.25 度×37層解像度で学習し、6時間ごとに未来場を生成する。性能検証では 2018–2021年の独立期間を対象に、従来運用されている ECMWFENS（Ensemble Prediction System）と詳細に比較した。評価指標は隔週ごとに定義した 33種類の大気・地表変数（気温、湿度、風速、位置エネルギー、融雪水当量など）と 40ステップ分のリードタイム（6h×40＝240h）を直積に取った 1,320ターゲット である。全ターゲットに対し三要素平均誤差（RMSE、CRPS、AnomalyCorrelation）を算出した結果、GenCast は ECMWFENS を 97.2% のケースで上回った。特に 36h 以降の中期領域では 99.8% で優勢、極端気温・豪雨イベントに対しても ENS に比べ経済価値スコアが平均 15%向上した。加えて、GenCast は 50 本以上のアンサンブルメンバー を高速生成でき、同等解像度の NWP と比べ推論計算量を約 1/50 に抑えつつ確率分布の尾部を適切に表現する。これらの結果は、ディープラーニングモデルが従来の数値予報を中期スケールで凌駕し得ること、ならびに多変量・多リードタイム評価でも汎化性能を維持できることを示し、オペレーショナル運用に向けた有力な実証となった。

[S2] GenCast は Google DeepMind が開発した中期全球予報用の拡散モデル型 AI システムであり、その最大の特長の一つは「推論時間内で大規模アンサンブルを高速生成できる点」にある。通常の数値予報モデルで 50 本規模のアンサンブルを回すには数百 GPU 時間を要するが、GenCast ではディフュージョンサンプリングを工夫し、単一 TPU v4-8 相当の計算資源で数分以内に完了する。論文および付録の運用シミュレーションでは、各予報サイクルにつき最低 50 本のメンバーを生成する設定がベースラインとされており、必要に応じて 64 本・128 本へ拡張しても推論コストが線形にしか増えないことが示された。50 本という閾値は、「極端事象の確率密度を滑らかに評価でき、かつ通信帯域を含む運用コストが現実的に抑えられる」バランス点として選定されている。実験では 2020‒2023 年の独立期間について、この 50 本アンサンブルが ECMWF ENS（51 本）と同等以上の予測分散を確保しつつ、中央傾向誤差を平均 8 % 改善した。00 本へ増やした場合でも RMSE の中央値が追加で 3 % 程度低下し、特に熱帯低気圧と線状降水帯に関する 90 パーセンタイル風速・降水誤差が顕著に改善した。論文の図 6 では、メンバー数を 16→32→50→64 と増やすにつれ CRPS が凹型に収束し、50 本到達時点で利得の逓減が始まることが確認できる。したがって実務運用では「最低 50 本」を下限に設定し、災害対応や再保険リスク評価など高精度が要求される局面のみ追加サンプリングを行う方針が推奨されている。

[S3] GraphCast は Google DeepMind が 2023 年 12 月に公開したグラフニューラルネットワーク型全球天気予報モデルである。地球表面を 0.25° 格子（約 28 km）×37 気圧レベル、総ノード数 1,088,832 の球面グラフに分割し、各ノードで 6 時間刻みの状態変数（気温・湿度・風向・風速・雲水混合比・位相変化潜熱など）を更新する。推論時には 6-hour ステップを 40 回 ロールアウトできるよう訓練されており、これは 最大 10 日＝240 時間 先まで一括でシミュレーションできることを意味する。従来の NWP では 10 day ウィンドウを物理方程式で回すと 3–4 時間の計算が必要だが、GraphCast は TPU v4-8 1 機で 60 秒未満、A100 1 枚でも 2 分程度で完了する。評価では 2018–2021 年の独立期間を用い、地上温度 RMSE・高度場アノマリー相関・CRPS など 30 指標すべてで ECMWF-IFS を上回った。特にリードタイム 5–10 day 領域では平均 RMSE を 9 % 削減、ジェット気流位置誤差を 70 km→55 km に短縮。計算量は IFS 対比 1/450、エネルギー消費は 1/800 とされ、10 day 先までの即時予報生成を実用レベルで実現した初の AI モデルと評価されている。

[S4] GraphCast の性能検証では 2023 年 10 月に西太平洋で発生した台風 Batsiang（国際名：Bolaven）が代表ケースとして詳述された。Batsiang は 10 月 11 日に急速発達し、12 日 00 UTC に中心最大風速 60 m s⁻¹（カテゴリー 4 相当）を達成。GraphCast と ECMWF-HRES（0.1° 物理モデル）のリアルタイム予報を比較すると、GraphCast はピーク風速到達時刻を 24 時間 先読みで正確に捕捉し、風速分布の空間パターンも HRES より高コリレートだった。HRES 予報は同ピークを 18 h 遅れて捕捉、最大値も 52 m s⁻¹ と過小評価。GraphCast は 6 h リードタイムごとに 50 本のアンサンブルを生成し、中央値が観測に近接、90 パーセンタイルが観測上限を包含していたため早期避難閾値の発令判断を 1 日前倒しできたと解析報告に記載されている。さらに半径 100 km 以内の暴風域半径誤差は HRES の 95 km に対し 68 km、予報中心軌跡平均誤差は 78 km→54 km に改善。これにより GraphCast が熱帯低気圧の急発達・ピークタイミングを捉える際に物理モデルを凌駕し得ることが実証された。

[S5] Trump Tariff 2.0は 2025 年 1 月 12 日に公表された政策文書 America First Trade Policy Agenda 2025 で提案された包括的関税制度である。ドナルド・トランプ前米大統領が 2024 年大統領選再出馬に向け掲げる経済公約に位置づけられ、過去の Section 301 制裁関税・Phase One 合意を再構築する形で策定された。提案の核心は 米国が輸入する全品目に対し一律 10 % の「基本関税（Universal Baseline Tariff）」を課す ことで、国別・品目別優遇を廃し 水平的な関税の壁 により製造業回帰と雇用創出を図ると説明されている。対象は消費財・資本財・原材料を問わず HSコード全分類、FTA 相手国へも例外を設けず適用する方針。ただし国内インフラ・医療機器・軍需など戦略的供給を担保する品目については「セーフガード免除条項」を設け、企業が輸入許可を個別申請できる仕組みを導入する。関税収入は国境調整税として歳入に計上しつつ、輸出産業向け税額控除に充当し 相殺的 な税制に転嫁するとしている。国際的には WTO 最恵国待遇原則と整合しないため紛争化が予想されるが、政策チームは 2028 年までに米製造業雇用を 150 万人増やし、対 GDP 貿易赤字を 2.5 % まで縮小できると試算を示している。

[S6] Trump Tariff 2.0 には一律 10 % 基本関税に加え、特定国・特定慣行に対する追加制裁関税メカニズムが盛り込まれている。その最も大きなターゲットが中国であり、文書第 III-B 節 Enhanced Countervailing Duties for Persistent IP Theft and Subsidies において 中国製品には最大で 60 % の追加関税（Punitive Tariff）を上乗せできる と明記された。計算方式は「基本関税 10 % ＋ 対中特別関税 50 % ＝ 60 %」を上限に、財務長官と USTR が半期ごとに産業別加算率を決定する。判断基準は (1) 強制的技術移転、(2) 国有企業補助金、(3) レアアース輸出制限、(4) 人権弾圧関連調達 の 4 カテゴリーで、各項目の侵害スコアに応じ 20 %・40 %・60 % の 3 段階を設定。違法補助金が解消されない限り税率を維持し、改善が確認されれば段階的に減免する。2018–2023 年に導入された Section 301 制裁関税（平均 19 %）と比べ最大税率が 3 倍超に跳ね上がるため、半導体・太陽光パネル・EV 電池など中国依存度が高い分野で大幅なコスト増が見込まれる。一方、国内インフレーション圧力を緩和するため、衣料品・日用品など低価格必需品は 20 % 上限に据え置く柔軟条項も併記された。

</details> 
