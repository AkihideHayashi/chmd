# chmd
chainer molecular dynamics

TODO: 
1.ANI1の先頭でinputをdirect座標として想定してcartesianに変換する処理を入れる。
  ついでにこの時に座標を単位格子に引き戻す。

2.データをdirect座標にとり直したpickleを作る。forceも同じく。(pickleを読んで座標と力に関して格子の逆行列をとるだけ。)
3.現状のMDはseries formを前提に書かれているが、これをparallel formにしたところで基本的には関係はない。
  ただし、nose-hooverだけは悪影響がありそう。
  nose-hooverはbatchformを使えばもっと簡単に書き直せるので、これを使って書き直す。しかし、基本的には探索にはnose-hooverはいらないので、優先度は低い。