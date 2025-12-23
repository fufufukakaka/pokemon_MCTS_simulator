:::message
本記事は株式会社ポケモンとは一切関係のない非公式のファンコンテンツです
:::

こんにちは。fufufukakaka です。
普段からポケモン対戦(シングルバトル)のデータを使って遊んでいます。
データで遊ぶのが主な趣味なのですが対戦自体も楽しませていただいており、SVは全シーズンを遊んでいます。一番好きなのはレギュレーションE、最高レートは1800をちょっと超えたくらいでエンジョイしています。

過去作をまとめたスライドはこちら↓

@[speakerdeck](09f80a2ee69b4936acd89e3153ac6299)

ポケモン対戦とは、の話などはきっと皆さんご存知かと思うので割愛いたします。

## なぜ対戦ボット？

AlphaGo などに代表されるように囲碁や将棋などといったゲームにおいて、機械学習で作ったボットが活躍するようになってそれなりの時間が経ちました。
しかし、これらは完全情報ゲームであって、ポケモンのような不完全情報ゲームとは異なり盤面からすべての情報を取得できます。相手が隠している情報は何もありません。
対してポケモンは不完全情報の塊のようなものです。同じく不完全情報ゲームに取り組んでいる事例として、鳥海先生らが立ち上げた[人狼知能](https://aiwolf.org/)が有名かと思われます。基本的なモチベーションは同じです。
ただし、ポケモンが明確に異なるのは手番です。人狼は本来どう喋るかにプロトコルなどありません。しかし、ポケモンは基本的に**同時手番の対戦ゲーム**です。なので、とっても解析がしやすい枠組みということですね。

さて、長くなりましたが要するに解析がしやすい枠組みだし、そもそも対戦ボットを作ることで「ポケモン対戦が強いってどういうことなんだろう？」を解き明かしたいのが自分の興味であります。あわよくばそれで強くなりたい。あわよくば...

## ポケモン対戦ボットの先行研究

既に幾つか先行研究があるので紹介します。

### 前提知識: PokemonShowdown

https://pokemonshowdown.com/

非公式の対戦サービスです。少し触るとわかりますが全世代分のポケモン対戦が実装されています。
対戦ログなども取得できる上、何より Web ベースで動いていて API も公開されているので世に出ているポケモン対戦ボットのほとんどが PokemonShowdown をベースにしています。
IMO としては画像とか音楽も結構そのまま使っているのであんまり近づかないようにしていますが、事例がほぼ全てこれを基盤にしているので紹介しました。

### FoulPlay

https://github.com/pmariglia/foul-play

ポケモン内では「イカサマ」の技を示す言葉です。
Pokemon Showdown で機能する MCTS(モンテカルロ木探索) ベースの手法を採用しています。
ただ MCTS を用いるとそれは完全情報ゲームであることを前提に置かないといけない(相手の手札からあり得る未来をシミュレートして一番良い現在の選択肢を割り出す)のですが、ポケモン対戦はそうではありません。
そこで FoulPlay では「ありそうな技構成をたくさんサンプリングして、それぞれの世界で MCTS」→「結果を重ね合わせて手を選ぶ」というデータを組み合わせた MCTS を行っています。これは PokemonShowdown 上にデータがたくさんあるから実現する手法といえますね。

### 強化学習系

#### poke-env

https://github.com/hsahovic/poke-env

強化学習の実行環境である Gymnasium 互換のライブラリです。これも PokemonShowdown を前提としています。
これ自体は手法を提供しているわけではなく強化学習で使える環境という感じなのですが、これをベースに提案された手法がいくつかあります。

#### Metamon(2025)

https://metamon.tech/

Pokémon Showdown の膨大な人間対戦ログから オフライン RL でエージェントを学習したものです。
学習したポリシーをオンラインで Showdown でのオンライン対戦環境に出して、人間と戦わせて評価しています。

### LLM系

#### PokeLLMon(2024)

https://poke-llm-on.github.io/

https://arxiv.org/abs/2402.01118

![](https://storage.googleapis.com/zenn-user-upload/2b238b191338-20251214.png)

GPT-4 などの LLM に対戦ログを与えたうえで次の行動を提案させる手法を提案しています。すごく強いわけではないですが、これだけでそこそこ対戦できるというのは面白かった。

#### PokeChamp(2025)

ICML 2025 Spotlight

https://sites.google.com/view/pokechamp-llm

PokéLLMon を発展させた LLM + ミニマックス探索エージェントです。
行動サンプリング・相手モデリング・評価関数に LLM を使いながら、ミニマックス木を読んで次アクションを提案する構造になっています。

## 新提案

いずれも地平を切り開いてくれた先行事例たちでした。これをベースに、新しい提案をしてみようと思います。

- 先行事例ではいずれも選出部分がおざなりになっている
  - PokéLLMon は6体がランダムに決まってそれらを全部使う方式特化
  - PokeChamp は6体の内どの3体を出すかは(多分)決め打ち(あんまり書いてなかった)
  - Metamon は第四世代を前提としている。この時代はどうやら 6→3 ではなかったらしく、6体全部を使う想定
- できれば Pokemon Showdown は使いたくない
  - そもそも実対戦環境とはかなり乖離している(気がする)。ダブル(VGC)はまだしも、シングルは構築を組んでいる途中の練習場みたいな扱いなような。なので、そこから得られる対戦ログはあまり良質ではないのでは？という仮説
  - 権利的にもできるだけ距離を取りたい
- 不完全情報を扱う、というところにもっと焦点を当てたい
  - 相手の持ち物はなんだろう、努力値はなんだろう、という仮説を置いて、観測された情報でそれを更新して勝ち筋を見定めるのが本質なんじゃないか？と思っているのでそれに近い手法にしたい
 
ということで、選出をしっかり考えたモデルと、不完全情報を組み込んだ強化学習エージェントを組み合わせるべきだなと思いました。

## 選出提案モデル

実はこれはもう取り組んだことがありまして、それを紹介します。

@[speakerdeck](93a830f5cd2b4d9abe36e55696e27632)

概要としては、https://sv.pokedb.tokyo/guide/opendata にて公開されているパーティを使って、Masked Language Model でポケモンを単語とみなした BERT を学習させます。構築のうち1匹が欠けているとして、この一匹を推定しよう、という枠組みですね。

![](https://storage.googleapis.com/zenn-user-upload/b4ffd1304e68-20251217.png)

で、作った BERT を用いて相手パーティとこちらパーティをくっつけたポケモンの並びに対して Token Classification を行うというものです。固有表現認識と同じ枠組みにすると、推論回数が一回で済むので便利！そのうえ、意外といい感じに推論できるので満足しました。

## 公開されている構築情報を使って、自己対戦ログだけで対戦行動を予測するボットを作る

これで選出はできるようになりました。次はいよいよ実際に対戦させるところです。

### 公開情報の収集

まずは対戦させるパーティの情報が必要です。[ポケモンデータベース](https://sv.pokedb.tokyo/)では上位構築が公開されていますが、ここで公開されているのはパーティの6匹とその技構成程度です。努力値や持ち物、テラスタイプなどの詳細情報は、リンク先の構築記事を見に行かないとわかりません。

構築記事の形式はまちまちで、[バトメモ](https://play.google.com/store/apps/details?id=com.kuranezumi.BattleMemories&hl=ja)などを使って1枚の画像にまとめている場合も多いです。こうした画像形式のものは諦めて、テキストで詳細を書いてくれている記事のみを対象としました。これらの記事から LLM (GPT-5 mini) を使って6体の詳細情報を抽出しています。

### 対戦シミュレート環境

得られた上位構築を対戦させる環境を用意しました。ポケモンSVにおける対戦要素（ダメージ計算、特性処理、テラスタル等）を可能な限り再現したものです。

もともとは[こちらのブログ](https://hfps4469.hatenablog.com/entry/2024/09/29/200912#%E8%A1%8C%E5%8B%95%E3%81%A8%E6%96%B9%E7%AD%96%E9%96%A2%E6%95%B0)で公開されていたコードを拝借し、修正しつつ今回の対戦環境としました。自分で修正した威嚇の特性処理がその後も間違っていることにしばらく気付かなくて、何回か学習させたモデルが無に帰す光を浴びてしまいました。

### 対戦エンジン: ReBeL

対戦で使われる AI エンジンには ReBeL を採用しました。これは Meta が発表したポーカー用 AI で、**不完全情報を扱うために「信念 (Belief)」という概念を導入している**のが特徴です。

ポーカーでは相手の手札がわかりません。同様に、ポケモン対戦でも相手の持ち物や技構成、テラスタイプなどは最初はわかりません。ReBeL はこうした隠された情報に対して確率分布（信念）を持ち、対戦中に得られる情報でその分布をベイズ更新していきます。

今回のポケモン対戦で扱う信念は以下の通りです。

#### 相手ポケモンの「型」に対する信念

相手の各ポケモンについて、以下の情報の組み合わせを「型仮説」として扱います：

| 情報 | 説明 | 観測による更新 |
|------|------|----------------|
| **相手の選出したポケモン** | 3体 | 交代などで判明 |
| **技構成** | 4つの技 | 技使用時に判明 |
| **持ち物** | きあいのタスキ、こだわりスカーフ等 | 持ち物発動時に判明 |
| **テラスタイプ** | テラスタル時のタイプ | テラスタル使用時に判明 |
| **特性** | いかく、ばけのかわ等 | 特性発動時に判明 |
| **努力値配分** | AS振り、HB振り等 | 性格から推定 |

例えば「ハバタクカミ」について、対戦開始時点では以下のような型仮説が考えられます：

- **スカーフ型**: ムーンフォース/シャドーボール/サイコショック/マジカルフレイム、こだわりスカーフ、テラスフェアリー
- **眼鏡型**: ムーンフォース/シャドーボール/サイコショック/エナジーボール、こだわりメガネ、テラスフェアリー
- **襷型**: ムーンフォース/シャドーボール/でんじは/ちょうはつ、きあいのタスキ、テラスゴースト
- ...

これらの型仮説それぞれに確率を割り当て、対戦中の観測で更新していきます。

なお、初期仮説はシーズン37の Pokemon Home から得られる統計値を使うことにしました。なのでびっくりテラスには負けやすいのですが、それはしょうがない...

#### 持ち物の判明イベント

タスキとかたべのこしとかですね。それらとタイミングが違うのは風船やブーストエナジーのように場に出たときに即座に消費やログが出る系で、これも途中まで無視してしまっていた...

#### 特性の判明イベント

- **場に出た時**: いかく、ひでり、あめふらし、エレキメイカー、おみとおし 等
- **ダメージを受けた時**: がんじょう、ばけのかわ、マルチスケイル、もらいび 等
- **ターン終了時**: かそく、ポイズンヒール、ムラっけ 等
- **パラドックス系**: こだいかっせい、クォークチャージ...

#### 信念の更新例

具体例で説明します。相手のハバタクカミが場に出てきたとします。

1. **初期状態**: 各型仮説に事前確率を割り当て（使用率データから算出）
2. **ムーンフォースを使用**: 「ムーンフォースを持たない型」の確率が0に → 残りで正規化
3. **こちらより先に行動**: スカーフの確率が上昇（素早さから推定）
4. **テラスタルでフェアリーに**: テラスフェアリー以外の確率が0に

擬似コードで表すと以下のようになります：

```python
class BeliefState:
    """相手ポケモンの型に対する信念状態"""

    def __init__(self, pokemon_name: str, usage_db: UsageDatabase):
        # 使用率データから型仮説と事前確率を生成
        self.hypotheses: dict[TypeHypothesis, float] = {}
        for type_data in usage_db.get_types(pokemon_name):
            hypothesis = TypeHypothesis(
                moves=type_data.moves,      # 例: ["ムーンフォース", "シャドーボール", ...]
                item=type_data.item,        # 例: "こだわりスカーフ"
                tera_type=type_data.tera,   # 例: "フェアリー"
                ability=type_data.ability,  # 例: "こだいかっせい"
            )
            self.hypotheses[hypothesis] = type_data.usage_rate

        self._normalize()

    def update(self, observation: Observation):
        """観測に基づいてベイズ更新"""
        for hypothesis in self.hypotheses:
            if not self._is_consistent(hypothesis, observation):
                # 観測と矛盾する仮説の確率を0に
                self.hypotheses[hypothesis] = 0.0

        self._normalize()

    def _is_consistent(self, hypothesis: TypeHypothesis, obs: Observation) -> bool:
        """仮説が観測と矛盾しないかチェック"""
        match obs.type:
            case ObservationType.MOVE_USED:
                # 使用された技を持っていない仮説は矛盾
                return obs.move_name in hypothesis.moves

            case ObservationType.ITEM_REVEALED:
                # 判明した持ち物と異なる仮説は矛盾
                return hypothesis.item == obs.item_name

            case ObservationType.TERASTALLIZED:
                # テラスタイプが異なる仮説は矛盾
                return hypothesis.tera_type == obs.tera_type

            case ObservationType.ABILITY_REVEALED:
                # 特性が異なる仮説は矛盾
                return hypothesis.ability == obs.ability_name

        return True

    def _normalize(self):
        """確率の正規化（合計を1にする）"""
        total = sum(self.hypotheses.values())
        if total > 0:
            for h in self.hypotheses:
                self.hypotheses[h] /= total

# 使用例
belief = BeliefState("ハバタクカミ", usage_db)
# → {"スカーフ型": 0.35, "眼鏡型": 0.30, "襷型": 0.20, ...}

belief.update(Observation(type=MOVE_USED, move_name="ムーンフォース"))
# → ムーンフォースを持たない型の確率が0に、残りで正規化

belief.update(Observation(type=TERASTALLIZED, tera_type="フェアリー"))
# → テラスフェアリー以外の確率が0に
# → {"スカーフ型": 0.54, "眼鏡型": 0.46, ...}  # 襷型は消えた
```

#### CFR (Counterfactual Regret Minimization) による戦略計算

信念状態が定まったら、次は「どの手を打つか」を決める必要があります。ReBeL では CFR というアルゴリズムを使ってナッシュ均衡に近い戦略を計算します。

簡単に言うと、「相手がどの型であっても、平均的に最も良い結果が得られる行動」を選ぶということです。相手の型が確定していない状況でも、複数の型仮説をサンプリングして平均を取ることで、堅実な判断ができます。

CFR "後悔"(Regret) を軸に考えます。各行動について「もしこの行動を選んでいたら、どれだけ良かった(悪かった)か」を計算し、後悔が大きい行動ほど次回選びやすくなるよう戦略を更新していきます。

```python
class CFRSolver:
    """Counterfactual Regret Minimization による戦略計算"""

    def __init__(self):
        # 累積後悔: 各行動について「選んでいれば得られた追加利得」の合計
        self.cumulative_regret: dict[Action, float] = defaultdict(float)
        # 累積戦略: 各イテレーションで使用した戦略の合計（平均戦略の計算用）
        self.cumulative_strategy: dict[Action, float] = defaultdict(float)

    def solve(self, battle_state: BattleState, belief: BeliefState, iterations: int = 100):
        """CFRを実行して戦略を計算"""
        for _ in range(iterations):
            # 信念から「ありえる世界」をサンプリング
            sampled_world = belief.sample_world()

            # 現在の戦略を後悔値から計算（Regret Matching）
            current_strategy = self._regret_matching()

            # 各行動の期待利得を計算
            action_values = {}
            for action in battle_state.legal_actions():
                # この行動を取った場合の期待勝率を計算
                action_values[action] = self._evaluate_action(
                    battle_state, sampled_world, action
                )

            # 現在の戦略に従った期待利得
            strategy_value = sum(
                current_strategy[a] * action_values[a]
                for a in action_values
            )

            # 後悔の更新: 各行動について「選んでいればどれだけ得したか」
            for action, value in action_values.items():
                regret = value - strategy_value
                self.cumulative_regret[action] += regret

            # 累積戦略の更新（平均戦略の計算用）
            for action, prob in current_strategy.items():
                self.cumulative_strategy[action] += prob

        # 最終的な戦略は累積戦略の平均
        return self._get_average_strategy()

    def _regret_matching(self) -> dict[Action, float]:
        """
        Regret Matching: 正の後悔に比例した確率で行動を選択

        後悔が大きい = 「選んでおけばよかった」行動
        → 次はその行動を選びやすくする
        """
        positive_regrets = {
            a: max(0, r) for a, r in self.cumulative_regret.items()
        }
        total = sum(positive_regrets.values())

        if total > 0:
            # 正の後悔に比例した確率
            return {a: r / total for a, r in positive_regrets.items()}
        else:
            # 全て非正なら一様分布
            actions = list(self.cumulative_regret.keys())
            return {a: 1.0 / len(actions) for a in actions}

    def _evaluate_action(
        self, state: BattleState, world: SampledWorld, action: Action
    ) -> float:
        """
        行動の価値を評価

        実際の実装では、ここでゲーム木を探索するか、
        学習済みのValue Networkを使って評価する
        """
        next_state = state.apply_action(action, world)
        return self.value_network(next_state)  # 勝率予測

    def _get_average_strategy(self) -> dict[Action, float]:
        """累積戦略から平均戦略を計算"""
        total = sum(self.cumulative_strategy.values())
        if total > 0:
            return {a: c / total for a, c in self.cumulative_strategy.items()}
        return {}


# 使用例
solver = CFRSolver()
belief = BeliefState("ハバタクカミ", usage_db)

# 100回のイテレーションで戦略を計算
strategy = solver.solve(battle_state, belief, iterations=100)
# → {"ムーンフォース": 0.45, "シャドーボール": 0.30, "交代:カイリュー": 0.25}
```

この「後悔を最小化する」というアプローチを繰り返すと、理論的にはナッシュ均衡（お互いに最善を尽くしている状態）に収束することが証明されています。ポーカーAI の Libratus や Pluribus もこの CFR を基盤としています。

ポケモン対戦への適用では、相手の型が不確実な状況でも「どの型であっても大きく損しない」行動を選べるようになります。例えば、相手がスカーフ型かもしれないし眼鏡型かもしれない場合、どちらに対してもそこそこ有効な行動を選ぶ、といった判断ができます。いわゆる安定択ということですね。

### 強化学習による学習

ここまでで ReBeL の仕組みを説明しました。次は実際に学習させていきます。

#### 学習の流れ

学習は自己対戦（Self-Play）ベースの強化学習で行います。AlphaZero と同様のアプローチです。

![](https://storage.googleapis.com/zenn-user-upload/106a86912096-20251218.png)

1. **Self-Play**: 現在のモデル同士で対戦し、(盤面, 行動, 勝敗) のデータを収集
2. **学習**: 収集したデータで Value Network と Selection BERT を更新
3. **評価**: 新しいモデルが前のモデルより強くなったか確認
4. **繰り返し**: 1-3 を繰り返してモデルを強化

#### 学習するネットワーク

学習対象は2つあります。

**1. Value Network（勝率予測）**

盤面の状態から「この状況での勝率」を予測するネットワークです。CFR の `_evaluate_action` で使用され、各行動の価値を評価します。

```python
class ValueNetwork(nn.Module):
    """盤面状態から勝率を予測"""

    def forward(self, battle_state: Tensor) -> float:
        # 盤面の特徴量（HP、状態異常、場の状態など）を入力
        # 0.0〜1.0 の勝率を出力
        return win_probability
```

**2. Selection BERT（選出予測）**

前述の選出提案モデルです。Self-Play の結果から「勝った試合での選出」を学習することで、より強い選出ができるようになります。

#### ロスカーブ

実際に100イテレーション学習させた結果がこちらです。

![](https://storage.googleapis.com/zenn-user-upload/836b929d6364-20251218.png)

Value Network のロスは順調に下がっており、勝敗予測の精度が向上していることがわかります。(流石に滑らかに減少とはいかなかった)

![](https://storage.googleapis.com/zenn-user-upload/fd0bcdd5129e-20251218.png)

Selection BERT も同様にロスが下がっています。ただし、こちらは途中から下げ止まる傾向がありました。選出にも択があるのでむずかしい

### 評価実験

学習させたモデルの強さを確認するため、いくつかのベースラインと対戦させました。

#### 対戦相手

1. **ランダム**: 合法手からランダムに選択
2. **CFR のみ**: Value Network を使わず、ロールアウトで評価する素朴な CFR

これを、学習済 ReBel と50回対戦させました。どちらも上位構築からランダムに構築を選択してきます。ReBel と CFR の信念情報はシーズン 37 の Pokemon Home 情報を用いました。
なお、学習済 ReBel は必ず Selection BERT を使いますが、相手は使う場合と使わない場合とで比較しました。

#### 結果

| 対戦カード | ReBel の勝率 |
|-----------|------|
| Selection BERT + ReBeL vs 選出ランダム・ランダム行動 | 100% |
| Selection BERT + ReBeL vs Selection BERT・ランダム行動 | 90% |
| Selection BERT + ReBeL vs 選出ランダム・CFRのみ | 78% |
| Selection BERT + ReBeL vs Selection BERT・CFRのみ | 90% |

ランダム行動のときのみですが、相手が Selection BERT を使ってくると勝率が下がっているのがわかるかと思います。CFR は頑張られてしまったのですが、こちらは行動で巻き返せるので選出の影響は小さかったのかも。ランダムの方はダイレクトに選出が勝率に影響したんじゃないかなと思います。

ということで、だいたい動く対戦ボットができました！...たぶん。

## 対戦インターフェースもつくってみた

せっかく学習させたので、実際に対戦できるインターフェースも作りました。
そもそもちゃんと対戦できるか不安だったし...

![](https://storage.googleapis.com/zenn-user-upload/98a2d2401227-20251218.png)

![](https://storage.googleapis.com/zenn-user-upload/39e2c607bf4c-20251218.png)

#### 機能

**1. 勝率の可視化**

Value Network が予測する勝率をリアルタイムで表示します。自分の行動によって勝率がどう変わるかを確認できます。

```
現在の勝率: 62.3%

行動ごとの勝率変化予測:
  ムーンフォース  → 65.1% (+2.8%)
  シャドーボール  → 58.7% (-3.6%)
  交代: カイリュー → 71.2% (+8.9%)  ← 最善手
```

**2. 最適行動の表示**

CFR が計算した戦略（各行動の選択確率）を表示します。「この場面では交代が最善」といった判断の根拠がわかります。

**3. 信念状態の表示**

相手の型に対する現在の信念（確率分布）を表示します。「相手のハバタクカミはスカーフ型が54%、眼鏡型が46%」のような情報が見られます。

### 何度か対戦してみた

![](https://storage.googleapis.com/zenn-user-upload/14457a6dcee9-20251218.png)

コライドン - 連撃ウーラオス対面。連撃ウーラオス は h12a220b252d4s20 でかなり硬い。
相手の裏にはルギアとキラフロルがいる状況。こちらは鉢巻けたぐりをしたところ、そのターンから一気に相手の方での交代択(ルギア・キラフロル引き)が強くなった。かなり硬いウーラオスに良いダメージが入ったので鉢巻と判断された？
(なぜか水テラスアクアジェットをされたので、なんかまだバグがある気がする)

![](https://storage.googleapis.com/zenn-user-upload/8612fec23512-20251218.png)

自分: コライドン、アルセウス(電気)、サーフゴー、ランドロス、オーロンゲ、イーユイ
相手: アルセウス(フェアリー)・黒バド・ヘイラッシャ・ドオー・グライオン・メタモン

![](https://storage.googleapis.com/zenn-user-upload/95097705ed06-20251218.png)

コライドン、オーロンゲ、サーフゴー という選出をしたこちらに対して、相手の選出は
アルセウス(フェアリー)・黒バド・ドオー。なんか...結構嫌な選出をされてる気がする(たぶん)。

![](https://storage.googleapis.com/zenn-user-upload/a38f28ee1108-20251218.png)

ちなみに、何も事前知識を与えていないのにグライオンはしっかりと "みがまも" を学習しています。ここが一番感動しました。

## 今後やりたいこと

ということで、長文おつかれさまでした！ありがとうございます！
ここまででだいたい自分のやりたい対戦ボットを作ることができました。ただ、まだまだやりたいことがあります。その一部をここに future work として書いておきます。

#### 1. 人間との対戦評価

現状は AI 同士の対戦でしか評価していません。実際に人間のプレイヤーと対戦させて、どの程度通用するのかを確認したいです。
もうちょっとボットを強くしたうえで対戦してくれる人を募集...できたらうれしい

#### 2. 感想戦ツール

ポケモン対戦をやっていると、対戦後に「あの場面、何が最善だったんだろう？」と振り返りたくなることがあります。

学習させた ReBeL を使えば、各場面での最善手を確認できます。これを使った「感想戦ツール」を作れるかも？対戦ログと ReBel の予測結果を LLM に渡していい感じに反省したい。

## おわりに

長くなりましたが、ポケモン対戦 AI を ReBeL ベースで実装した話でした。

- 不完全情報を「信念」として扱い、ベイズ更新で絞り込んでいく
- CFR でナッシュ均衡に近い戦略を計算する
- Self-Play で Value Network と Selection BERT を学習させる

SVランクマッチ最終(?)シーズン頑張りましょう〜
