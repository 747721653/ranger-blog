(window.webpackJsonp=window.webpackJsonp||[]).push([[10],{327:function(t,s,m){"use strict";m.r(s);var a=m(8),c=Object(a.a)({},(function(){var t=this,s=t._self._c;return s("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[s("h2",{attrs:{id:"什么是transformer"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#什么是transformer"}},[t._v("#")]),t._v(" 什么是Transformer")]),t._v(" "),s("p",[t._v("Transformer是一个基于"),s("strong",[t._v("纯注意力机制")]),t._v("的网络模型，它没有运用循环或是卷积神经网络")]),t._v(" "),s("p",[t._v("整体的结构为一个"),s("strong",[t._v("编码器-解码器")])]),t._v(" "),s("h2",{attrs:{id:"网络架构"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#网络架构"}},[t._v("#")]),t._v(" 网络架构")]),t._v(" "),s("p",[s("strong",[t._v("编码器-解码器结构")]),t._v("：")]),t._v(" "),s("p",[t._v("编码器会将序列中的每一个词表示成一个单独的向量（输出）；\n解码器拿到编码器的输出之后生成一个新的序列（和原始序列长度不一定相等），编码的时候可以一次性全部生成，\n而解码的时候只能够一个个来（自回归（过去时刻的输出可以看做当前时刻的输入）），\n解码第n个词的时候可以看到前n-1个词的信息，这里类似于rnn。")]),t._v(" "),s("div",{staticClass:"center-container"},[s("p",[s("img",{attrs:{src:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image-20230215205919963.2ojlk7kw0wa0.webp",alt:"image-20230215205919963"}})])]),s("p",[s("code",[t._v("Add&Norm")]),t._v("：在每一个子层后面都加上一个这个，为了保证输入输出一致，这里设定每一个层输出的维度为512")]),t._v(" "),s("p",[s("code",[t._v("Norm")]),t._v("：每个子层的输出为：")]),t._v(" "),s("p"),s("p",[s("mjx-container",{staticClass:"MathJax",attrs:{jax:"CHTML",display:"true"}},[s("mjx-math",{staticClass:"MJX-TEX",attrs:{display:"true"}},[s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"L"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"a"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"y"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"e"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"r"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"N"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"o"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"r"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"m"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n",attrs:{space:"3"}},[s("mjx-c",{attrs:{c:"+"}})],1),s("mjx-mi",{staticClass:"mjx-i",attrs:{space:"3"}},[s("mjx-c",{attrs:{c:"S"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"u"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"b"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"l"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"a"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"y"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"e"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"r"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1)],1)],1)],1),s("p"),t._v(" "),s("p",[s("code",[t._v("LayerNorm")]),t._v("：将单个样本（不同于"),s("code",[t._v("batchNorm")]),t._v("是对每个特征做正则化）内转换成均值为0，方差为1的格式（将每个值减去均值同时除以方差就行了）")]),t._v(" "),s("p",[s("code",[t._v("Add")]),t._v("：残差连接，作用有两个：")]),t._v(" "),s("ol",[s("li",[s("p",[t._v("降低模型复杂度，放置过拟合")])]),t._v(" "),s("li",[s("p",[t._v("防止梯度消失")])])]),t._v(" "),s("p",[t._v("没做残差连接的梯度"),s("mjx-container",{staticClass:"MathJax",attrs:{jax:"CHTML"}},[s("mjx-math",{staticClass:"MJX-TEX"},[s("mjx-mfrac",[s("mjx-frac",[s("mjx-num",[s("mjx-nstrut"),s("mjx-mrow",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2202"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"f"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"g"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1)],1)],1),s("mjx-dbox",[s("mjx-dtable",[s("mjx-line"),s("mjx-row",[s("mjx-den",[s("mjx-dstrut"),s("mjx-mrow",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2202"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1)],1)],1)],1)],1)],1)],1)],1),s("mjx-mo",{staticClass:"mjx-n",attrs:{space:"4"}},[s("mjx-c",{attrs:{c:"="}})],1),s("mjx-mfrac",{attrs:{space:"4"}},[s("mjx-frac",[s("mjx-num",[s("mjx-nstrut"),s("mjx-mrow",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2202"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"f"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"g"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1)],1)],1),s("mjx-dbox",[s("mjx-dtable",[s("mjx-line"),s("mjx-row",[s("mjx-den",[s("mjx-dstrut"),s("mjx-mrow",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2202"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"g"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1)],1)],1)],1)],1)],1)],1)],1),s("mjx-mo",{staticClass:"mjx-n",attrs:{space:"3"}},[s("mjx-c",{attrs:{c:"D7"}})],1),s("mjx-mfrac",{attrs:{space:"3"}},[s("mjx-frac",[s("mjx-num",[s("mjx-nstrut"),s("mjx-mrow",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2202"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"g"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1)],1)],1),s("mjx-dbox",[s("mjx-dtable",[s("mjx-line"),s("mjx-row",[s("mjx-den",[s("mjx-dstrut"),s("mjx-mrow",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2202"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1)],1)],1)],1)],1)],1)],1)],1)],1)],1),t._v("，网络层数很深之后容易导致梯度乘以接近0的数组从而让梯度消失。")],1),t._v(" "),s("p",[t._v("加了残差连接的梯度："),s("mjx-container",{staticClass:"MathJax",attrs:{jax:"CHTML"}},[s("mjx-math",{staticClass:"MJX-TEX"},[s("mjx-mfrac",[s("mjx-frac",[s("mjx-num",[s("mjx-nstrut"),s("mjx-mrow",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2202"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"f"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"g"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"+"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"g"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1)],1)],1),s("mjx-dbox",[s("mjx-dtable",[s("mjx-line"),s("mjx-row",[s("mjx-den",[s("mjx-dstrut"),s("mjx-mrow",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2202"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1)],1)],1)],1)],1)],1)],1)],1),s("mjx-mo",{staticClass:"mjx-n",attrs:{space:"4"}},[s("mjx-c",{attrs:{c:"="}})],1),s("mjx-mfrac",{attrs:{space:"4"}},[s("mjx-frac",[s("mjx-num",[s("mjx-nstrut"),s("mjx-mrow",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2202"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"f"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"g"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1)],1)],1),s("mjx-dbox",[s("mjx-dtable",[s("mjx-line"),s("mjx-row",[s("mjx-den",[s("mjx-dstrut"),s("mjx-mrow",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2202"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"g"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1)],1)],1)],1)],1)],1)],1)],1),s("mjx-mo",{staticClass:"mjx-n",attrs:{space:"3"}},[s("mjx-c",{attrs:{c:"D7"}})],1),s("mjx-mfrac",{attrs:{space:"3"}},[s("mjx-frac",[s("mjx-num",[s("mjx-nstrut"),s("mjx-mrow",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2202"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"g"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1)],1)],1),s("mjx-dbox",[s("mjx-dtable",[s("mjx-line"),s("mjx-row",[s("mjx-den",[s("mjx-dstrut"),s("mjx-mrow",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2202"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1)],1)],1)],1)],1)],1)],1)],1),s("mjx-mo",{staticClass:"mjx-n",attrs:{space:"3"}},[s("mjx-c",{attrs:{c:"+"}})],1),s("mjx-mfrac",{attrs:{space:"3"}},[s("mjx-frac",[s("mjx-num",[s("mjx-nstrut"),s("mjx-mrow",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2202"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"g"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1)],1)],1),s("mjx-dbox",[s("mjx-dtable",[s("mjx-line"),s("mjx-row",[s("mjx-den",[s("mjx-dstrut"),s("mjx-mrow",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2202"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1)],1)],1)],1)],1)],1)],1)],1)],1)],1),t._v("，加了一项且值相对较大，即使是很深的网络也不容易导致梯度消失")],1),t._v(" "),s("p",[t._v("解码器中的第一块"),s("img",{staticStyle:{zoom:"33%"},attrs:{src:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image.5mis7o8t8oo0.webp",alt:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image.5mis7o8t8oo0.webp"}}),t._v("：带掩码的注意力机制，保证在t时间不会看到t时间之后的哪些输入")]),t._v(" "),s("p",[s("strong",[t._v("Attention")]),t._v("：")]),t._v(" "),s("p",[t._v("Attention的原理：")]),t._v(" "),s("div",{staticClass:"center-container"},[s("p",[s("img",{attrs:{src:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image.5w9bcygwij80.webp",alt:"image"}})])]),s("p",[t._v("首先输出是三个v的累加，假如query和第一个key比较接近，那么输出的时候第一个value的权重就会比较大，远离第一个value的value权重就会慢慢变小")]),t._v(" "),s("p"),s("p",[s("mjx-container",{staticClass:"MathJax",attrs:{jax:"CHTML",display:"true"}},[s("mjx-math",{staticClass:"MJX-TEX",attrs:{display:"true"}},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"A"}}),s("mjx-c",{attrs:{c:"t"}}),s("mjx-c",{attrs:{c:"t"}}),s("mjx-c",{attrs:{c:"e"}}),s("mjx-c",{attrs:{c:"m"}}),s("mjx-c",{attrs:{c:"p"}}),s("mjx-c",{attrs:{c:"t"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2061"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"Q"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:","}})],1),s("mjx-mi",{staticClass:"mjx-i",attrs:{space:"2"}},[s("mjx-c",{attrs:{c:"K"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:","}})],1),s("mjx-mi",{staticClass:"mjx-i",attrs:{space:"2"}},[s("mjx-c",{attrs:{c:"V"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1),s("mjx-mo",{staticClass:"mjx-n",attrs:{space:"4"}},[s("mjx-c",{attrs:{c:"="}})],1),s("mjx-mi",{staticClass:"mjx-n",attrs:{space:"4"}},[s("mjx-c",{attrs:{c:"s"}}),s("mjx-c",{attrs:{c:"o"}}),s("mjx-c",{attrs:{c:"f"}}),s("mjx-c",{attrs:{c:"t"}}),s("mjx-c",{attrs:{c:"m"}}),s("mjx-c",{attrs:{c:"a"}}),s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2061"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mstyle",[s("mjx-mfrac",[s("mjx-frac",{attrs:{type:"d"}},[s("mjx-num",[s("mjx-nstrut",{attrs:{type:"d"}}),s("mjx-mrow",[s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"Q"}})],1),s("mjx-msup",[s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"K"}})],1),s("mjx-script",{staticStyle:{"vertical-align":"0.363em"}},[s("mjx-mi",{staticClass:"mjx-i",attrs:{size:"s"}},[s("mjx-c",{attrs:{c:"T"}})],1)],1)],1)],1)],1),s("mjx-dbox",[s("mjx-dtable",[s("mjx-line",{attrs:{type:"d"}}),s("mjx-row",[s("mjx-den",[s("mjx-dstrut",{attrs:{type:"d"}}),s("mjx-msqrt",[s("mjx-sqrt",[s("mjx-surd",[s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"221A"}})],1)],1),s("mjx-box",{staticStyle:{"padding-top":"0.037em"}},[s("mjx-msub",[s("mjx-mi",{staticClass:"mjx-i",attrs:{noIC:"true"}},[s("mjx-c",{attrs:{c:"d"}})],1),s("mjx-script",{staticStyle:{"vertical-align":"-0.15em"}},[s("mjx-mi",{staticClass:"mjx-i",attrs:{size:"s"}},[s("mjx-c",{attrs:{c:"k"}})],1)],1)],1)],1)],1)],1)],1)],1)],1)],1)],1)],1)],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"V"}})],1)],1)],1)],1),s("p"),t._v(" "),s("p",[s("mjx-container",{staticClass:"MathJax",attrs:{jax:"CHTML"}},[s("mjx-math",{staticClass:"MJX-TEX"},[s("mjx-msub",[s("mjx-mi",{staticClass:"mjx-i",attrs:{noIC:"true"}},[s("mjx-c",{attrs:{c:"d"}})],1),s("mjx-script",{staticStyle:{"vertical-align":"-0.15em"}},[s("mjx-mi",{staticClass:"mjx-i",attrs:{size:"s"}},[s("mjx-c",{attrs:{c:"k"}})],1)],1)],1)],1)],1),t._v("指的是Q和K向量的维度，Q和K向量在编码器中的维度是一样的，长度可能不一，例如在解码器中由于Q是来自目标序列（即mask multi-head attention输出的），因此不能保证Q和K的长度一致，向量的矩阵计算之后，做softmax就得到了对应的概率，相加为1")],1),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image.j9p19dwnz9s.webp",alt:"image"}}),s("img",{attrs:{src:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image.64p6ber6yo80.webp",alt:"image"}}),s("img",{attrs:{src:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image.1zxtei3umpk0.webp",alt:"image"}})]),t._v(" "),s("div",{staticClass:"center-container"},[s("p",[s("img",{attrs:{src:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image.aqonb4ofykc.webp",alt:"image"}})])]),s("p",[t._v("之所以要除以"),s("mjx-container",{staticClass:"MathJax",attrs:{jax:"CHTML"}},[s("mjx-math",{staticClass:"MJX-TEX"},[s("mjx-msqrt",[s("mjx-sqrt",[s("mjx-surd",[s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"221A"}})],1)],1),s("mjx-box",{staticStyle:{"padding-top":"0.037em"}},[s("mjx-msub",[s("mjx-TeXAtom",[s("mjx-mi",{staticClass:"mjx-i",attrs:{noIC:"true"}},[s("mjx-c",{attrs:{c:"d"}})],1)],1),s("mjx-script",{staticStyle:{"vertical-align":"-0.15em"}},[s("mjx-TeXAtom",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"k"}})],1)],1)],1)],1)],1)],1)],1)],1)],1),t._v("，是为了防止向量做点积的时候"),s("mjx-container",{staticClass:"MathJax",attrs:{jax:"CHTML"}},[s("mjx-math",{staticClass:"MJX-TEX"},[s("mjx-msub",[s("mjx-mi",{staticClass:"mjx-i",attrs:{noIC:"true"}},[s("mjx-c",{attrs:{c:"d"}})],1),s("mjx-script",{staticStyle:{"vertical-align":"-0.15em"}},[s("mjx-mi",{staticClass:"mjx-i",attrs:{size:"s"}},[s("mjx-c",{attrs:{c:"k"}})],1)],1)],1)],1)],1),t._v("很长时某些值过大导致softmax时概率更趋于1从而导致梯度变得很小，训练不动")],1),t._v(" "),s("p",[t._v("一整个的计算流程如下：")]),t._v(" "),s("div",{staticClass:"center-container"},[s("p",[s("img",{attrs:{src:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image.5ykt9ooth4c0.webp",alt:"image"}})])]),s("p",[s("img",{staticStyle:{zoom:"50%"},attrs:{src:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image.2ecx2zr8lr6s.webp"}}),t._v("这里的Mask是可选的，只有在解码器的时候有用到，用处是防止模型关注当前t时刻后面的信息，具体做法是给t时刻后面的这一块计算结果换成一个非常大的负数，那么在softmax中的指数计算就会趋近于0，其权重就会变成0，在之后与V的计算时就相当于是不参与计算了")]),t._v(" "),s("p",[s("strong",[t._v("多头注意力机制")]),t._v("：")]),t._v(" "),s("div",{staticClass:"center-container"},[s("p",[s("img",{attrs:{src:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image.st8ia43n0ps.webp",alt:"image"}})])]),s("p",[t._v("之所以要使用多头注意力机制，是因为在原来单个注意力机制中，没有什么能够学习的参数，整个流程都是给定的，因此在多头注意力计算中，将V、K、Q做线性投影到低维，投影时给定的投影矩阵W是可以学习的，同时有h个这样的结构，相当于给了h次学习的机会，希望能够学到不一样的一些信息，最后拼接起来再投影回去")]),t._v(" "),s("div",{staticClass:"center-container"},[s("p",[s("img",{attrs:{src:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image.3m7sewtebjc0.webp",alt:"image"}})])]),s("div",{staticClass:"center-container"},[s("p",[s("img",{attrs:{src:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image.6oyee92eink0.webp",alt:"image"}})])]),s("p",[t._v("这里h=8，"),s("mjx-container",{staticClass:"MathJax",attrs:{jax:"CHTML"}},[s("mjx-math",{staticClass:"MJX-TEX"},[s("mjx-mstyle",[s("mjx-mspace",{staticStyle:{width:"0.222em"}})],1),s("mjx-msub",[s("mjx-mi",{staticClass:"mjx-i",attrs:{noIC:"true"}},[s("mjx-c",{attrs:{c:"d"}})],1),s("mjx-script",{staticStyle:{"vertical-align":"-0.15em"}},[s("mjx-TeXAtom",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"k"}})],1)],1)],1)],1),s("mjx-mo",{staticClass:"mjx-n",attrs:{space:"4"}},[s("mjx-c",{attrs:{c:"="}})],1),s("mjx-msub",{attrs:{space:"4"}},[s("mjx-mi",{staticClass:"mjx-i",attrs:{noIC:"true"}},[s("mjx-c",{attrs:{c:"d"}})],1),s("mjx-script",{staticStyle:{"vertical-align":"-0.15em"}},[s("mjx-TeXAtom",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"v"}})],1)],1)],1)],1),s("mjx-mo",{staticClass:"mjx-n",attrs:{space:"4"}},[s("mjx-c",{attrs:{c:"="}})],1),s("mjx-msub",{attrs:{space:"4"}},[s("mjx-mi",{staticClass:"mjx-i",attrs:{noIC:"true"}},[s("mjx-c",{attrs:{c:"d"}})],1),s("mjx-script",{staticStyle:{"vertical-align":"-0.15em"}},[s("mjx-TeXAtom",{attrs:{size:"s"}},[s("mjx-TeXAtom",[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"m"}})],1),s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"o"}})],1),s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"d"}})],1),s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"e"}})],1),s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"l"}})],1)],1)],1)],1)],1),s("mjx-TeXAtom",[s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"/"}})],1)],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"h"}})],1),s("mjx-TeXAtom",{attrs:{space:"4"}},[s("mjx-mover",[s("mjx-over",{staticStyle:{"padding-bottom":"0.2em","padding-left":"0.389em"}},[s("mjx-TeXAtom",{attrs:{size:"s"}})],1),s("mjx-base",[s("mjx-TeXAtom",[s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"="}})],1)],1)],1)],1)],1),s("mjx-mn",{staticClass:"mjx-n",attrs:{space:"4"}},[s("mjx-c",{attrs:{c:"6"}}),s("mjx-c",{attrs:{c:"4"}})],1)],1)],1)],1),t._v(" "),s("h2",{attrs:{id:"模型的大体结构"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#模型的大体结构"}},[t._v("#")]),t._v(" 模型的大体结构")]),t._v(" "),s("div",{staticClass:"center-container"},[s("p",[s("img",{attrs:{src:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image.3qyixiugt3q0.webp",alt:"image"}})])]),s("p",[t._v("之所以是自注意力机制，是因为在进入多头注意力层的时候，同样的向量复制成了三份，分别作为K、Q、V。")]),t._v(" "),s("p",[t._v("假定输入为长度为n的一个序列，向量化之后变为矩阵大小为n*d，输出大小也一致，输出的每一个向量都是输入的所有向量的加权累加（就是softmax那一步乘以V向量）")]),t._v(" "),s("p",[t._v("解码器第一个子层的输出会被当做Q向量输入进第二个子层，其输入是另一个序列（机器翻译中的翻译结果）")]),t._v(" "),s("p",[t._v("Transformer模型在训练和投入使用时的模式是不一样的，训练时由于知道目标输出序列，因此做mask后可以并行进行计算，但实际使用时是一个个单词输出的，每预测一次都会加大解码器输入序列的长度（开始的时候是0？）")]),t._v(" "),s("p",[s("strong",[t._v("Feed-forward（前馈神经网络）")]),t._v("：")]),t._v(" "),s("p",[t._v("具体公式描述："),s("mjx-container",{staticClass:"MathJax",attrs:{jax:"CHTML"}},[s("mjx-math",{staticClass:"MJX-TEX"},[s("mjx-mi",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"F"}}),s("mjx-c",{attrs:{c:"F"}}),s("mjx-c",{attrs:{c:"N"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"2061"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1),s("mjx-mo",{staticClass:"mjx-n",attrs:{space:"4"}},[s("mjx-c",{attrs:{c:"="}})],1),s("mjx-mo",{staticClass:"mjx-n",attrs:{space:"4"}},[s("mjx-c",{attrs:{c:"m"}}),s("mjx-c",{attrs:{c:"a"}}),s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"("}})],1),s("mjx-mn",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"0"}})],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:","}})],1),s("mjx-mi",{staticClass:"mjx-i",attrs:{space:"2"}},[s("mjx-c",{attrs:{c:"x"}})],1),s("mjx-msub",[s("mjx-mi",{staticClass:"mjx-i",attrs:{noIC:"true"}},[s("mjx-c",{attrs:{c:"W"}})],1),s("mjx-script",{staticStyle:{"vertical-align":"-0.15em"}},[s("mjx-mn",{staticClass:"mjx-n",attrs:{size:"s"}},[s("mjx-c",{attrs:{c:"1"}})],1)],1)],1),s("mjx-mo",{staticClass:"mjx-n",attrs:{space:"3"}},[s("mjx-c",{attrs:{c:"+"}})],1),s("mjx-msub",{attrs:{space:"3"}},[s("mjx-mi",{staticClass:"mjx-i",attrs:{noIC:"true"}},[s("mjx-c",{attrs:{c:"b"}})],1),s("mjx-script",{staticStyle:{"vertical-align":"-0.15em"}},[s("mjx-mn",{staticClass:"mjx-n",attrs:{size:"s"}},[s("mjx-c",{attrs:{c:"1"}})],1)],1)],1),s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:")"}})],1),s("mjx-msub",[s("mjx-mi",{staticClass:"mjx-i",attrs:{noIC:"true"}},[s("mjx-c",{attrs:{c:"W"}})],1),s("mjx-script",{staticStyle:{"vertical-align":"-0.15em"}},[s("mjx-mn",{staticClass:"mjx-n",attrs:{size:"s"}},[s("mjx-c",{attrs:{c:"2"}})],1)],1)],1),s("mjx-mo",{staticClass:"mjx-n",attrs:{space:"3"}},[s("mjx-c",{attrs:{c:"+"}})],1),s("mjx-msub",{attrs:{space:"3"}},[s("mjx-mi",{staticClass:"mjx-i",attrs:{noIC:"true"}},[s("mjx-c",{attrs:{c:"b"}})],1),s("mjx-script",{staticStyle:{"vertical-align":"-0.15em"}},[s("mjx-mn",{staticClass:"mjx-n",attrs:{size:"s"}},[s("mjx-c",{attrs:{c:"2"}})],1)],1)],1)],1)],1)],1),t._v(" "),s("p",[t._v("包含两个线性变换和一个Relu激活，transformer在这边对每个向量都会用相同的参数应用一次，之所以不对整个矩阵作用，是因为之前的Attention已经把每个向量的信息提取出来并aggregation了")]),t._v(" "),s("p",[t._v("内层维度为"),s("mjx-container",{staticClass:"MathJax",attrs:{jax:"CHTML"}},[s("mjx-math",{staticClass:"MJX-TEX"},[s("mjx-msub",[s("mjx-mi",{staticClass:"mjx-i",attrs:{noIC:"true"}},[s("mjx-c",{attrs:{c:"d"}})],1),s("mjx-script",{staticStyle:{"vertical-align":"-0.15em"}},[s("mjx-TeXAtom",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"m"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"o"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"d"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"e"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"l"}})],1)],1)],1)],1)],1)],1),t._v("的4倍，在这里就是2048，外层又映射回512")],1),t._v(" "),s("p",[t._v("空间复杂度为n*dff，dff = 4*"),s("mjx-container",{staticClass:"MathJax",attrs:{jax:"CHTML"}},[s("mjx-math",{staticClass:"MJX-TEX"},[s("mjx-msub",[s("mjx-mi",{staticClass:"mjx-i",attrs:{noIC:"true"}},[s("mjx-c",{attrs:{c:"d"}})],1),s("mjx-script",{staticStyle:{"vertical-align":"-0.15em"}},[s("mjx-TeXAtom",{attrs:{size:"s"}},[s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"m"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"o"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"d"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"e"}})],1),s("mjx-mi",{staticClass:"mjx-i"},[s("mjx-c",{attrs:{c:"l"}})],1)],1)],1)],1)],1)],1),t._v("，因为要保存每个向量的计算结果")],1),t._v(" "),s("p",[s("strong",[t._v("Embedding")]),t._v("：")]),t._v(" "),s("p",[t._v("就是将词映射为一个向量，这里将权重乘了一个"),s("mjx-container",{staticClass:"MathJax",attrs:{jax:"CHTML"}},[s("mjx-math",{staticClass:"MJX-TEX"},[s("mjx-msqrt",[s("mjx-sqrt",[s("mjx-surd",[s("mjx-mo",{staticClass:"mjx-n"},[s("mjx-c",{attrs:{c:"221A"}})],1)],1),s("mjx-box",{staticStyle:{"padding-top":"0.037em"}},[s("mjx-msub",[s("mjx-mi",{staticClass:"mjx-i",attrs:{noIC:"true"}},[s("mjx-c",{attrs:{c:"d"}})],1),s("mjx-script",{staticStyle:{"vertical-align":"-0.15em"}},[s("mjx-TeXAtom",{attrs:{size:"s"}},[s("mjx-TeXAtom",[s("mjx-mtext",{staticClass:"mjx-mit"},[s("mjx-c",{attrs:{c:"m"}}),s("mjx-c",{attrs:{c:"o"}}),s("mjx-c",{attrs:{c:"d"}}),s("mjx-c",{attrs:{c:"e"}}),s("mjx-c",{attrs:{c:"l"}})],1)],1)],1)],1)],1)],1)],1)],1)],1)],1)],1),t._v(" "),s("p",[s("strong",[t._v("Positional Encoding")]),t._v("："),s("strong",[t._v("加入时序信息")])]),t._v(" "),s("p",[t._v("首先将位置信息encoding，然后把它加进encoding后的原始数据中")]),t._v(" "),s("p",[t._v("将词所在位置信息encoding进原始信息中")]),t._v(" "),s("p",[s("img",{staticStyle:{zoom:"67%"},attrs:{src:"https://cdn.statically.io/gh/747721653/picx-images-hosting@master/image.42w80q2xf500.webp"}}),t._v(":第一个线性层是一个简单的全连接神经网络，它将解码器输出的向量投影到一个更大的向量中，相当于是单词表吧，每个向量的位置代表一个单词，之后经过一个softmax输出每个单词的概率，概率最高的作为模型的输出。")])])}),[],!1,null,null,null);s.default=c.exports}}]);