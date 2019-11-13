# chmd
chainer molecular dynamics

TODO: 


3.現状のMDはseries formを前提に書かれているが、これをparallel formにしたところで基本的には関係はない。
  ただし、nose-hooverだけは悪影響がありそう。
  nose-hooverはbatchformを使えばもっと簡単に書き直せるので、これを使って書き直す。しかし、基本的には探索にはnose-hooverはいらないので、優先度は低い。


  普通のvelocity verletに関しても、現状のbatchというのはあまりにも味気ないので、もうちょっとaseよりにしてもいいかも。
  device structを改造して、init_scope機能をつけて、あとはインターフェースを実装する。
  MDの実装に応じて適切なclassを作ることが出来るように、できればBatchの実装は一番最後に回したい。
  Batchはevalを持っていた方がいいかも。

  chainerの真似をすることを考えると、batchはエネルギーを返し、それを外側でbackwardして、更新することになる。
  しかし、chainerへの不満として、できればgrad_evaであって欲しかったという思いがある。
  batchはevalによって現在の情報から




  ようわからんけど、今のBatch構造体というのは結構いいかも？

  nose-hooverのことを思うと、出来ればparallel形式の座標の生成はmdが責任を持ちたい。
  隣接の計算省略はevaluatorが責任を持ちたい。
  しかし、点電荷拡張を考えるとプロパティはevaluatorが責任を持ちたい。
  電荷分布生成を考えると、evaluatorも座標を拡張したい。
  流石に電荷分布は別の形で持った方がいいかも。


  ってか、parallel形式なんだから、熱浴変数は毎回concatでよくね？
  決めました。熱浴変数はmdが持ちます。基本的なものはbatchが持ちます。
  evaluatorはbatchを受け取ります。


  とりあえず、batchはただの構造体である現状を維持する？


  Batchはinterfaceを作って、OptBatchとMDBatchのinterfaceも作る。
  ANI1もinterfaceを作る。
  Neighbor listをMDが持つ。
  Neighbor listはbatchを受け取って、neighbor listを返すクラス。





4.そのうち、ANI1が原子毎のエネルギー分散を計算するようにする。
現状、reporterにorder of symbolsを渡しているが、これは結構気持ちわるい。
