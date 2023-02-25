(window.webpackJsonp=window.webpackJsonp||[]).push([[12],{329:function(t,a,r){"use strict";r.r(a);var s=r(7),v=Object(s.a)({},(function(){var t=this,a=t._self._c;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h2",{attrs:{id:"说明"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#说明"}},[t._v("#")]),t._v(" 说明")]),t._v(" "),a("p",[a("strong",[t._v("BatchNorm")]),t._v("：对一个batch-size样本内的每个特征做归一化")]),t._v(" "),a("p",[a("strong",[t._v("LayerNorm")]),t._v("：针对每条样本，对每条样本的所有特征做归一化")]),t._v(" "),a("h2",{attrs:{id:"举例"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#举例"}},[t._v("#")]),t._v(" 举例")]),t._v(" "),a("p",[t._v("假设现在有个二维矩阵，行代表batch-size，列代表样本特征")]),t._v(" "),a("ul",[a("li",[t._v("BatchNorm就是对这个二维矩阵中每一列的特征做归一化，即竖着做归一化")]),t._v(" "),a("li",[t._v("LayerNorm就是对这个二维矩阵中每一行数据做归一化，即横着做归一化")])]),t._v(" "),a("h2",{attrs:{id:"异同点"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#异同点"}},[t._v("#")]),t._v(" 异同点")]),t._v(" "),a("h3",{attrs:{id:"相同点"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#相同点"}},[t._v("#")]),t._v(" 相同点")]),t._v(" "),a("p",[a("strong",[t._v("都是在深度学习中让当前层的参数稳定下来，避免梯度消失或者梯度爆炸，方便后面的继续学习")])]),t._v(" "),a("h3",{attrs:{id:"不同点"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#不同点"}},[t._v("#")]),t._v(" 不同点")]),t._v(" "),a("ul",[a("li",[t._v("如果你的特征依赖不同样本的统计参数，那BatchNorm更有效， 因为它不考虑不同特征之间的大小关系，但是保留不同样本间的大小关系")]),t._v(" "),a("li",[t._v("Nlp领域适合用LayerNorm， CV适合BatchNorm")]),t._v(" "),a("li",[t._v("对于Nlp来说，它不考虑不同样本间的大小关系，保留样本内不同特征之间的大小关系")])])])}),[],!1,null,null,null);a.default=v.exports}}]);