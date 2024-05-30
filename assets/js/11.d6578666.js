(window.webpackJsonp=window.webpackJsonp||[]).push([[11],{329:function(t,a,s){"use strict";s.r(a);var r=s(8),e=Object(r.a)({},(function(){var t=this,a=t._self._c;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("p",[t._v("论文地址："),a("a",{attrs:{href:"https://arxiv.org/abs/2405.15370",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://arxiv.org/abs/2405.15370"),a("OutboundLink")],1)]),t._v(" "),a("h2",{attrs:{id:"亮点"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#亮点"}},[t._v("#")]),t._v(" 亮点")]),t._v(" "),a("ul",[a("li",[t._v("利用大语言模型对时间序列进行异常检测")]),t._v(" "),a("li",[t._v("不需要对大语言模型进行微调")]),t._v(" "),a("li",[t._v("设计了一个用大模型进行时间序列异常检测的完整流程")])]),t._v(" "),a("p",[t._v("总之，值得一看")]),t._v(" "),a("h2",{attrs:{id:"摘要"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#摘要"}},[t._v("#")]),t._v(" 摘要")]),t._v(" "),a("p",[t._v("时间序列异常检测（TSAD）通过识别偏离标准趋势的非典型模式，从而保持系统完整性并实现快速响应措施，在各个行业中发挥着至关重要的作用。传统的TSAD模型通常依赖于深度学习，需要大量的训练数据，并且作为黑匣子运行，缺乏对检测到的异常的可解释性。为了应对这些挑战，我们提出了LLMAD，这是一种新的TSAD方法，它使用大型语言模型（LLM）来提供准确和可解释的TSAD结果。LLMAD通过检索正和负相似的时间序列片段，创新性地将LLM应用于上下文异常检测，显著提高了LLM的有效性。此外，LLMAD采用异常检测思想链（AnoCoT）方法来模拟其决策过程的专家逻辑。这种方法进一步提高了其性能，并使LLMAD能够通过多角度为其检测提供解释，这对用户决策尤为重要。在三个数据集上的实验表明，我们的LLMAD实现了与最先进的深度学习方法相当的检测性能，同时为检测提供了显著的可解释性。据我们所知，这是TSAD首次直接使用LLM的工作。")]),t._v(" "),a("h2",{attrs:{id:"方法"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#方法"}},[t._v("#")]),t._v(" 方法")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/747721653/picx-images-hosting@master/paper/image.4ckpw739br.webp",alt:"image"}})]),t._v(" "),a("h3",{attrs:{id:"时间序列数据预处理"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#时间序列数据预处理"}},[t._v("#")]),t._v(" 时间序列数据预处理")]),t._v(" "),a("p",[t._v("为了让大模型更易分析输入的时间序列数据，作者首先对输入的时间序列进行了预处理")]),t._v(" "),a("p",[t._v("预处理的步骤如下：")]),t._v(" "),a("ul",[a("li",[t._v("重缩放（Rescaling）：将序列的数值转换成大语言模型易于理解的范围")]),t._v(" "),a("li",[t._v("设置索引（Indexing）：按照时间顺序为每个时间点设置索引，用于之后异常点的报告与分析")])]),t._v(" "),a("h3",{attrs:{id:"时间序列情景学习-time-series-icl"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#时间序列情景学习-time-series-icl"}},[t._v("#")]),t._v(" 时间序列情景学习（Time Series ICL）")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/747721653/picx-images-hosting@master/paper/0bfa4a9a285bb0de30098f7d6aa7636d.361enlxyzq.webp",alt:"0bfa4a9a285bb0de30098f7d6aa7636d"}})]),t._v(" "),a("p",[t._v("首先将已有的时间序列数据分为正常数据库以及异常数据库，对于输入的时间序列，通过FastDTW算法计算它在两个数据库中最相似的其他正常或异常序列，并将它们输出并排列，以为大模型提供正常或异常数据的参考。")]),t._v(" "),a("h3",{attrs:{id:"异常检测思维链-anocot"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#异常检测思维链-anocot"}},[t._v("#")]),t._v(" 异常检测思维链（AnoCoT）")]),t._v(" "),a("p",[t._v("用CoT提示技术指导大语言模型逐步推理。原文提示模板如下：")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/747721653/picx-images-hosting@master/paper/image.4g4btxgfzh.webp",alt:"image"}})]),t._v(" "),a("p",[t._v("其中引入了领域专家知识（异常类型定义、异常级别定义等内容）、思维链提示、情景提示（之前检索的数据）")]),t._v(" "),a("h3",{attrs:{id:"可解释的时间序列异常检测"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#可解释的时间序列异常检测"}},[t._v("#")]),t._v(" 可解释的时间序列异常检测")]),t._v(" "),a("p",[t._v("让大语言模型生成全面的异常检测报告，从"),a("strong",[t._v("异常解释")]),t._v("、"),a("strong",[t._v("异常类型")]),t._v("、"),a("strong",[t._v("警告等级")]),t._v("这三个方面进行。")])])}),[],!1,null,null,null);a.default=e.exports}}]);