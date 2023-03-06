(window.webpackJsonp=window.webpackJsonp||[]).push([[28],{345:function(t,a,s){"use strict";s.r(a);var n=s(7),e=Object(n.a)({},(function(){var t=this,a=t._self._c;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h2",{attrs:{id:"tensordataset类"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#tensordataset类"}},[t._v("#")]),t._v(" TensorDataset类")]),t._v(" "),a("p",[t._v("TensorDataset能够用来对tensor进行打包，包装成Dataset")]),t._v(" "),a("div",{staticClass:"language-python line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[t._v("torch_dataset "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" Data"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("TensorDataset"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" y"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])]),t._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[t._v("1")]),a("br")])]),a("h2",{attrs:{id:"dataset类"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#dataset类"}},[t._v("#")]),t._v(" Dataset类")]),t._v(" "),a("p",[t._v("pytorch中所有的数据都是Dataset的子类，我们在使用pytorch生成训练数据时，可以创建一个继承Dataset类的数据类来方便我们使用数据")]),t._v(" "),a("p",[t._v("在创建Dataset子类时，需要重写getitem方法和len方法，其中getitem方法返回输入索引后对应的特征和标签，而len方法则返回数据集的总数据个数")]),t._v(" "),a("p",[t._v("在init初始化过程中，我们也可以输入可代表数据集基本属性的相关内容，包括数据集的特征、标签、大小等信息")]),t._v(" "),a("p",[t._v("示例：")]),t._v(" "),a("div",{staticClass:"center-container"},[a("p",[a("img",{attrs:{src:"https://cdn.staticaly.com/gh/747721653/image-store@master/pytorch/image.714yiayvcs00.jpg",alt:"image"}})])]),a("p",[t._v("另外，我们能够使用在torch.utils.data中的random_split函数去切分数据集")]),t._v(" "),a("div",{staticClass:"center-container"},[a("p",[a("img",{attrs:{src:"https://cdn.staticaly.com/gh/747721653/image-store@master/pytorch/image.7a06ftd026c0.jpg",alt:"image"}})])]),a("p",[t._v("此时切分的结果是一个映射式的对象，只有dataset和indices两个属性，其中dataset属性用于查看原数据集对象，indices属性用于查看切分后数据集的每一条数据的index（序号）")]),t._v(" "),a("div",{staticClass:"center-container"},[a("p",[a("img",{attrs:{src:"https://cdn.staticaly.com/gh/747721653/image-store@master/pytorch/image.4yv62f50y080.jpg",alt:"image"}})])]),a("p",[t._v("此外，如果想从已建立的Dataset中划分训练集测试集，可以使用Subset进行拆分，示例如下：")]),t._v(" "),a("div",{staticClass:"language-python line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[t._v("train_dataset "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("utils"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("data"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("Subset"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("dataset"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("range")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("int")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("split_rate "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" dataset"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("lens"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\ntest_dataset "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("utils"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("data"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("Subset"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("dataset"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("range")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("int")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("split_rate "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" dataset"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("lens"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" dataset"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("lens"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])]),t._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[t._v("1")]),a("br"),a("span",{staticClass:"line-number"},[t._v("2")]),a("br")])]),a("h2",{attrs:{id:"dataloader类"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#dataloader类"}},[t._v("#")]),t._v(" DataLoader类")]),t._v(" "),a("p",[t._v("DataLoader类负责进行数据转化，将一般数据状态转换为“可建模”的状态，即不仅包含数据原始的数据信息，还包含数据处理方法信息，如调用几个线程进行训练、分多少批次等")]),t._v(" "),a("p",[t._v("原型：")]),t._v(" "),a("div",{staticClass:"language-python line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[t._v("torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("utils"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("data"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("DataLoader"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("dataset"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" batch_size"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" shuffle"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("False")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" sampler"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("None")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" batch_sampler"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("None")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" num_workers"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" collate_fn"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("None")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" pin_memory"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("False")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" drop_last"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("False")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" timeout"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" worker_init_fn"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("None")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" multiprocessing_context"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("None")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" generator"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("None")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" prefetch_factor"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" persistent_workers"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("False")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])]),t._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[t._v("1")]),a("br")])]),a("p",[t._v("几个重要参数：")]),t._v(" "),a("ul",[a("li",[t._v("batch_size：每次迭代输入多少数据")]),t._v(" "),a("li",[t._v("shuffle：是否需要先打乱顺序然后再进行小批量的切分，一般训练集需要乱序，而测试集乱序没有意义")]),t._v(" "),a("li",[t._v("num_worker：启动多少线程进行计算")])]),t._v(" "),a("p",[t._v("DataLoader的使用需要搭配迭代器来使用，"),a("code",[t._v("for i, (input, target) in enumerate(data_loader)")]),t._v("或"),a("code",[t._v("next(iter(data_loader))")]),t._v("都可以返回数据和标签")])])}),[],!1,null,null,null);a.default=e.exports}}]);