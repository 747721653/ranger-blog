(window.webpackJsonp=window.webpackJsonp||[]).push([[12],{329:function(t,a,s){"use strict";s.r(a);var n=s(7),e=Object(n.a)({},(function(){var t=this,a=t._self._c;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h2",{attrs:{id:"pytorch"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#pytorch"}},[t._v("#")]),t._v(" pytorch")]),t._v(" "),a("h3",{attrs:{id:"torch-triu"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#torch-triu"}},[t._v("#")]),t._v(" torch.triu()")]),t._v(" "),a("p",[t._v("返回一个上三角矩阵，常见于mask操作")]),t._v(" "),a("p",[t._v("函数原型：")]),t._v(" "),a("div",{staticClass:"language-python line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("def")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("triu")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("input")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" Tensor"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" diagonal"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" _int"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" out"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" Optional"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("Tensor"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("None")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v(" Tensor"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("\n")])]),t._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[t._v("1")]),a("br")])]),a("p",[t._v("示例："),a("a",{attrs:{href:"https://blog.csdn.net/weixin_39574469/article/details/118195536",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://blog.csdn.net/weixin_39574469/article/details/118195536"),a("OutboundLink")],1)]),t._v(" "),a("h3",{attrs:{id:"transpose"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#transpose"}},[t._v("#")]),t._v(" .transpose()")]),t._v(" "),a("p",[t._v("将tensor的维度进行交换")]),t._v(" "),a("p",[t._v("示例："),a("a",{attrs:{href:"https://blog.csdn.net/a250225/article/details/102636425",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://blog.csdn.net/a250225/article/details/102636425"),a("OutboundLink")],1)]),t._v(" "),a("h3",{attrs:{id:"model-eval"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#model-eval"}},[t._v("#")]),t._v(" model.eval()")]),t._v(" "),a("p",[t._v("在对模型进行评估时，需要使用这个函数")]),t._v(" "),a("p",[t._v("它的作用是不启用 Batch Normalization 和 Dropout")]),t._v(" "),a("p",[t._v("如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。\nmodel.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。\n对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。")]),t._v(" "),a("p",[t._v("教程："),a("a",{attrs:{href:"https://blog.csdn.net/lgzlgz3102/article/details/115987271",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://blog.csdn.net/lgzlgz3102/article/details/115987271"),a("OutboundLink")],1)]),t._v(" "),a("h3",{attrs:{id:"torch-linspace"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#torch-linspace"}},[t._v("#")]),t._v(" torch.linspace()")]),t._v(" "),a("p",[t._v("函数原型：")]),t._v(" "),a("div",{staticClass:"language-python line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[t._v("torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("linspace"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("start"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" end"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" steps"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("100")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" out"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("None")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" dtype"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("None")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" layout"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v("torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("strided"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" device"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("None")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" requires_grad"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("False")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" → Tensor\n")])]),t._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[t._v("1")]),a("br")])]),a("p",[t._v("作用为返回一个一维的tensor，包含从start到end的等距的steps个数据点")]),t._v(" "),a("p",[t._v("示例："),a("a",{attrs:{href:"https://blog.csdn.net/weixin_43255962/article/details/84347726",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://blog.csdn.net/weixin_43255962/article/details/84347726"),a("OutboundLink")],1)]),t._v(" "),a("h3",{attrs:{id:"torch-tensor-permute"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#torch-tensor-permute"}},[t._v("#")]),t._v(" torch.Tensor.permute()")]),t._v(" "),a("p",[t._v("将tensor的维度换位")]),t._v(" "),a("p",[t._v("参数："),a("code",[t._v("dims (int …*)")]),t._v(" 换位顺序")]),t._v(" "),a("p",[t._v("例：")]),t._v(" "),a("div",{staticClass:"language-python line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">>")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v(" x "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("randn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("3")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("5")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" \n"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">>")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v(" x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("size"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" \ntorch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("Size"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("3")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("5")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" \n"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">>")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v(" x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("permute"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("size"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" \ntorch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("Size"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("5")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("3")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])]),t._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[t._v("1")]),a("br"),a("span",{staticClass:"line-number"},[t._v("2")]),a("br"),a("span",{staticClass:"line-number"},[t._v("3")]),a("br"),a("span",{staticClass:"line-number"},[t._v("4")]),a("br"),a("span",{staticClass:"line-number"},[t._v("5")]),a("br")])]),a("p",[t._v("详解："),a("a",{attrs:{href:"https://blog.csdn.net/qq_43489708/article/details/125154452",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://blog.csdn.net/qq_43489708/article/details/125154452"),a("OutboundLink")],1)]),t._v(" "),a("h2",{attrs:{id:"numpy"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#numpy"}},[t._v("#")]),t._v(" numpy")]),t._v(" "),a("h3",{attrs:{id:"np-append"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#np-append"}},[t._v("#")]),t._v(" np.append()")]),t._v(" "),a("p",[t._v("函数原型：")]),t._v(" "),a("div",{staticClass:"language-python line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("def")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("append")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("arr"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" values"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" axis"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("None")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("\n")])]),t._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[t._v("1")]),a("br")])]),a("p",[a("strong",[t._v("参数：")])]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("arr")]),t._v(":需要被添加values的数组")]),t._v(" "),a("li",[a("strong",[t._v("values")]),t._v(":添加到数组arr中的值（array_like，类数组）")]),t._v(" "),a("li",[a("strong",[t._v("axis")]),t._v(":可选参数，如果axis没有给出，那么arr，values都将先展平成一维数组。注：如果axis被指定了，那么arr和values需要同为一维数组或者有相同的shape，否则报错：ValueError: arrays must have same number of dimensions")])]),t._v(" "),a("p",[t._v("示例："),a("a",{attrs:{href:"https://blog.csdn.net/weixin_42216109/article/details/93889047",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://blog.csdn.net/weixin_42216109/article/details/93889047"),a("OutboundLink")],1)]),t._v(" "),a("h3",{attrs:{id:"np-reshape"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#np-reshape"}},[t._v("#")]),t._v(" np.reshape()")]),t._v(" "),a("p",[t._v("改变nparray的维度大小")]),t._v(" "),a("p",[t._v("-1表示我不关心这个维度的大小，reshape(-1, 1)代表一列n行")]),t._v(" "),a("p",[t._v("示例："),a("a",{attrs:{href:"https://blog.csdn.net/qq_43511299/article/details/117259662",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://blog.csdn.net/qq_43511299/article/details/117259662"),a("OutboundLink")],1)]),t._v(" "),a("h2",{attrs:{id:"pandas"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#pandas"}},[t._v("#")]),t._v(" pandas")])])}),[],!1,null,null,null);a.default=e.exports}}]);