(window.webpackJsonp=window.webpackJsonp||[]).push([[92],{409:function(s,t,a){"use strict";a.r(t);var n=a(7),e=Object(n.a)({},(function(){var s=this,t=s._self._c;return t("ContentSlotsDistributor",{attrs:{"slot-key":s.$parent.slotKey}},[t("h2",{attrs:{id:"_1-静态资源"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_1-静态资源"}},[s._v("#")]),s._v(" 1. 静态资源")]),s._v(" "),t("h3",{attrs:{id:"_1-默认规则"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_1-默认规则"}},[s._v("#")]),s._v(" 1. 默认规则")]),s._v(" "),t("h4",{attrs:{id:"_1-静态资源映射"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_1-静态资源映射"}},[s._v("#")]),s._v(" 1. 静态资源映射")]),s._v(" "),t("p",[s._v("静态资源映射规则在 WebMvcAutoConfiguration 中进行了定义：")]),s._v(" "),t("ol",[t("li",[s._v("/webjars/** 的所有路径 资源都在 classpath:/META-INF/resources/webjars/")]),s._v(" "),t("li",[s._v("/** 的所有路径 资源都在 classpath:/META-INF/resources/、classpath:/resources/、classpath:/static/、classpath:/public/")]),s._v(" "),t("li",[s._v("所有静态资源都定义了缓存规则。【浏览器访问过一次，就会缓存一段时间】，但此功能参数无默认值\n"),t("ol",[t("li",[s._v("period： 缓存间隔。 默认 0S；")]),s._v(" "),t("li",[s._v("cacheControl：缓存控制。 默认无；")]),s._v(" "),t("li",[s._v("useLastModified：是否使用lastModified头。 默认 false；")])])])]),s._v(" "),t("h4",{attrs:{id:"_2-静态资源缓存"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_2-静态资源缓存"}},[s._v("#")]),s._v(" 2. 静态资源缓存")]),s._v(" "),t("p",[s._v("如前面所述")]),s._v(" "),t("ol",[t("li",[t("p",[s._v("所有静态资源都定义了缓存规则。【浏览器访问过一次，就会缓存一段时间】，但此功能参数无默认值")]),s._v(" "),t("ol",[t("li",[s._v("period： 缓存间隔。 默认 0S；")]),s._v(" "),t("li",[s._v("cacheControl：缓存控制。 默认无；")]),s._v(" "),t("li",[s._v("useLastModified：是否使用lastModified头。 默认 false；")])])])]),s._v(" "),t("h4",{attrs:{id:"_3-欢迎页"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_3-欢迎页"}},[s._v("#")]),s._v(" 3. 欢迎页")]),s._v(" "),t("p",[s._v("欢迎页规则在 WebMvcAutoConfiguration 中进行了定义：")]),s._v(" "),t("ol",[t("li",[s._v("在"),t("strong",[s._v("静态资源")]),s._v("目录下找 index.html")]),s._v(" "),t("li",[s._v("没有就在 templates下找index模板页")])]),s._v(" "),t("h4",{attrs:{id:"_4-favicon"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_4-favicon"}},[s._v("#")]),s._v(" 4. Favicon")]),s._v(" "),t("p",[s._v("在静态资源目录下找 favicon.ico")]),s._v(" "),t("p",[s._v("再给服务器发请求的时候还默认会请求图标")]),s._v(" "),t("h3",{attrs:{id:"_2-自定义静态资源规则"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_2-自定义静态资源规则"}},[s._v("#")]),s._v(" 2. 自定义静态资源规则")]),s._v(" "),t("blockquote",[t("p",[s._v("自定义静态资源路径、自定义缓存规则")])]),s._v(" "),t("h4",{attrs:{id:"_1-配置方式"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_1-配置方式"}},[s._v("#")]),s._v(" 1. 配置方式")]),s._v(" "),t("p",[t("code",[s._v("spring.mvc")]),s._v("： 静态资源访问前缀路径")]),s._v(" "),t("p",[t("code",[s._v("spring.web")]),s._v("：")]),s._v(" "),t("ul",[t("li",[s._v("静态资源目录")]),s._v(" "),t("li",[s._v("静态资源缓存策略")])]),s._v(" "),t("div",{staticClass:"language-properties line-numbers-mode"},[t("pre",{pre:!0,attrs:{class:"language-properties"}},[t("code",[t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#1、spring.web：")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("# 1.配置国际化的区域信息")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("# 2.静态资源策略(开启、处理链、缓存)")]),s._v("\n\n"),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#开启静态资源映射规则")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token key attr-name"}},[s._v("spring.web.resources.add-mappings")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token value attr-value"}},[s._v("true")]),s._v("\n\n"),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#设置缓存")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token key attr-name"}},[s._v("spring.web.resources.cache.period")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token value attr-value"}},[s._v("3600")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("##缓存详细合并项控制，覆盖period配置：")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("## 浏览器第一次请求服务器，服务器告诉浏览器此资源缓存7200秒，7200秒以内的所有此资源访问不用发给服务器请求，7200秒以后发请求给服务器")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token key attr-name"}},[s._v("spring.web.resources.cache.cachecontrol.max-age")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token value attr-value"}},[s._v("7200")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("## 共享缓存")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token key attr-name"}},[s._v("spring.web.resources.cache.cachecontrol.cache-public")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token value attr-value"}},[s._v("true")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#使用资源 last-modified 时间，来对比服务器和浏览器的资源是否相同没有变化。相同返回 304")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token key attr-name"}},[s._v("spring.web.resources.cache.use-last-modified")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token value attr-value"}},[s._v("true")]),s._v("\n\n"),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#自定义静态资源文件夹位置")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token key attr-name"}},[s._v("spring.web.resources.static-locations")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token value attr-value"}},[s._v("classpath:/a/,classpath:/b/,classpath:/static/")]),s._v("\n\n"),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#2、 spring.mvc")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("## 2.1. 自定义webjars路径前缀")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token key attr-name"}},[s._v("spring.mvc.webjars-path-pattern")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token value attr-value"}},[s._v("/wj/**")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("## 2.2. 静态资源访问路径前缀")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token key attr-name"}},[s._v("spring.mvc.static-path-pattern")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token value attr-value"}},[s._v("/static/**")]),s._v("\n")])]),s._v(" "),t("div",{staticClass:"line-numbers-wrapper"},[t("span",{staticClass:"line-number"},[s._v("1")]),t("br"),t("span",{staticClass:"line-number"},[s._v("2")]),t("br"),t("span",{staticClass:"line-number"},[s._v("3")]),t("br"),t("span",{staticClass:"line-number"},[s._v("4")]),t("br"),t("span",{staticClass:"line-number"},[s._v("5")]),t("br"),t("span",{staticClass:"line-number"},[s._v("6")]),t("br"),t("span",{staticClass:"line-number"},[s._v("7")]),t("br"),t("span",{staticClass:"line-number"},[s._v("8")]),t("br"),t("span",{staticClass:"line-number"},[s._v("9")]),t("br"),t("span",{staticClass:"line-number"},[s._v("10")]),t("br"),t("span",{staticClass:"line-number"},[s._v("11")]),t("br"),t("span",{staticClass:"line-number"},[s._v("12")]),t("br"),t("span",{staticClass:"line-number"},[s._v("13")]),t("br"),t("span",{staticClass:"line-number"},[s._v("14")]),t("br"),t("span",{staticClass:"line-number"},[s._v("15")]),t("br"),t("span",{staticClass:"line-number"},[s._v("16")]),t("br"),t("span",{staticClass:"line-number"},[s._v("17")]),t("br"),t("span",{staticClass:"line-number"},[s._v("18")]),t("br"),t("span",{staticClass:"line-number"},[s._v("19")]),t("br"),t("span",{staticClass:"line-number"},[s._v("20")]),t("br"),t("span",{staticClass:"line-number"},[s._v("21")]),t("br"),t("span",{staticClass:"line-number"},[s._v("22")]),t("br"),t("span",{staticClass:"line-number"},[s._v("23")]),t("br"),t("span",{staticClass:"line-number"},[s._v("24")]),t("br"),t("span",{staticClass:"line-number"},[s._v("25")]),t("br")])]),t("h4",{attrs:{id:"_2-代码方式"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_2-代码方式"}},[s._v("#")]),s._v(" 2. 代码方式")]),s._v(" "),t("blockquote",[t("ul",[t("li",[s._v("容器中只要有一个 WebMvcConfigurer 组件。配置的底层行为都会生效")]),s._v(" "),t("li",[s._v("@EnableWebMvc //禁用boot的默认配置")])])]),s._v(" "),t("div",{staticClass:"language-java line-numbers-mode"},[t("pre",{pre:!0,attrs:{class:"language-java"}},[t("code",[t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("//@EnableWebMvc //禁用boot的默认配置")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token annotation punctuation"}},[s._v("@Configuration")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("//这是一个配置类")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("public")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("class")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("MyConfig")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("implements")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("WebMvcConfigurer")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("{")]),s._v("\n\n\n    "),t("span",{pre:!0,attrs:{class:"token annotation punctuation"}},[s._v("@Override")]),s._v("\n    "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("public")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("void")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("addResourceHandlers")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("ResourceHandlerRegistry")]),s._v(" registry"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("{")]),s._v("\n        "),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("//保留以前规则，只要不加@EnableWebMvc注解，不写这个依然有默认配置")]),s._v("\n        "),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("WebMvcConfigurer")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("super")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("addResourceHandlers")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),s._v("registry"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(";")]),s._v("\n        "),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("//自己写新的规则。")]),s._v("\n        registry"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("addResourceHandler")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"/static/**"')]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v("\n                "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("addResourceLocations")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"classpath:/a/"')]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"classpath:/b/"')]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v("\n                "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("setCacheControl")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("CacheControl")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("maxAge")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token number"}},[s._v("1180")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("TimeUnit")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),t("span",{pre:!0,attrs:{class:"token constant"}},[s._v("SECONDS")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(";")]),s._v("\n    "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("}")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("}")]),s._v("\n")])]),s._v(" "),t("div",{staticClass:"line-numbers-wrapper"},[t("span",{staticClass:"line-number"},[s._v("1")]),t("br"),t("span",{staticClass:"line-number"},[s._v("2")]),t("br"),t("span",{staticClass:"line-number"},[s._v("3")]),t("br"),t("span",{staticClass:"line-number"},[s._v("4")]),t("br"),t("span",{staticClass:"line-number"},[s._v("5")]),t("br"),t("span",{staticClass:"line-number"},[s._v("6")]),t("br"),t("span",{staticClass:"line-number"},[s._v("7")]),t("br"),t("span",{staticClass:"line-number"},[s._v("8")]),t("br"),t("span",{staticClass:"line-number"},[s._v("9")]),t("br"),t("span",{staticClass:"line-number"},[s._v("10")]),t("br"),t("span",{staticClass:"line-number"},[s._v("11")]),t("br"),t("span",{staticClass:"line-number"},[s._v("12")]),t("br"),t("span",{staticClass:"line-number"},[s._v("13")]),t("br"),t("span",{staticClass:"line-number"},[s._v("14")]),t("br"),t("span",{staticClass:"line-number"},[s._v("15")]),t("br")])]),t("p",[t("strong",[s._v("下面这种写法也行")])]),s._v(" "),t("div",{staticClass:"language-java line-numbers-mode"},[t("pre",{pre:!0,attrs:{class:"language-java"}},[t("code",[t("span",{pre:!0,attrs:{class:"token annotation punctuation"}},[s._v("@Configuration")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("//这是一个配置类,给容器中放一个 WebMvcConfigurer 组件，就能自定义底层")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("public")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("class")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("MyConfig")]),s._v("  "),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("/*implements WebMvcConfigurer*/")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("{")]),s._v("\n\n    "),t("span",{pre:!0,attrs:{class:"token annotation punctuation"}},[s._v("@Bean")]),s._v("\n    "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("public")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("WebMvcConfigurer")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("webMvcConfigurer")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("{")]),s._v("\n        "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("return")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("new")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("WebMvcConfigurer")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("{")]),s._v("\n            "),t("span",{pre:!0,attrs:{class:"token annotation punctuation"}},[s._v("@Override")]),s._v("\n            "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("public")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("void")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("addResourceHandlers")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("ResourceHandlerRegistry")]),s._v(" registry"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("{")]),s._v("\n                registry"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("addResourceHandler")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"/static/**"')]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v("\n                        "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("addResourceLocations")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"classpath:/a/"')]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"classpath:/b/"')]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v("\n                        "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("setCacheControl")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("CacheControl")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("maxAge")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token number"}},[s._v("1180")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("TimeUnit")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),t("span",{pre:!0,attrs:{class:"token constant"}},[s._v("SECONDS")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(";")]),s._v("\n            "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("}")]),s._v("\n        "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("}")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(";")]),s._v("\n    "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("}")]),s._v("\n\n"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("}")]),s._v("\n")])]),s._v(" "),t("div",{staticClass:"line-numbers-wrapper"},[t("span",{staticClass:"line-number"},[s._v("1")]),t("br"),t("span",{staticClass:"line-number"},[s._v("2")]),t("br"),t("span",{staticClass:"line-number"},[s._v("3")]),t("br"),t("span",{staticClass:"line-number"},[s._v("4")]),t("br"),t("span",{staticClass:"line-number"},[s._v("5")]),t("br"),t("span",{staticClass:"line-number"},[s._v("6")]),t("br"),t("span",{staticClass:"line-number"},[s._v("7")]),t("br"),t("span",{staticClass:"line-number"},[s._v("8")]),t("br"),t("span",{staticClass:"line-number"},[s._v("9")]),t("br"),t("span",{staticClass:"line-number"},[s._v("10")]),t("br"),t("span",{staticClass:"line-number"},[s._v("11")]),t("br"),t("span",{staticClass:"line-number"},[s._v("12")]),t("br"),t("span",{staticClass:"line-number"},[s._v("13")]),t("br"),t("span",{staticClass:"line-number"},[s._v("14")]),t("br"),t("span",{staticClass:"line-number"},[s._v("15")]),t("br"),t("span",{staticClass:"line-number"},[s._v("16")]),t("br")])]),t("h2",{attrs:{id:"_2-路径匹配"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_2-路径匹配"}},[s._v("#")]),s._v(" 2. 路径匹配")]),s._v(" "),t("blockquote",[t("p",[t("strong",[s._v("Spring5.3")]),s._v(" 之后加入了更多的"),t("strong",[s._v("请求路径匹配")]),s._v("的实现策略；")]),s._v(" "),t("p",[s._v("以前只支持 "),t("strong",[s._v("AntPathMatcher")]),s._v(" 策略, 现在提供了 "),t("strong",[s._v("PathPatternParser")]),s._v(" 策略。并且可以让我们指定到底使用那种策略。")])]),s._v(" "),t("h3",{attrs:{id:"_1-ant风格路径用法"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_1-ant风格路径用法"}},[s._v("#")]),s._v(" 1. Ant风格路径用法")]),s._v(" "),t("p",[s._v("Ant 风格的路径模式语法具有以下规则：")]),s._v(" "),t("ul",[t("li",[s._v("*：表示"),t("strong",[s._v("任意数量")]),s._v("的字符。")]),s._v(" "),t("li",[s._v("?：表示任意"),t("strong",[s._v("一个字符")]),s._v("。")]),s._v(" "),t("li",[s._v("**：表示"),t("strong",[s._v("任意数量的目录")]),s._v("。")]),s._v(" "),t("li",[s._v("{}：表示一个命名的模式"),t("strong",[s._v("占位符")]),s._v("。")]),s._v(" "),t("li",[s._v("[]：表示"),t("strong",[s._v("字符集合")]),s._v("，例如[a-z]表示小写字母。")])]),s._v(" "),t("p",[s._v("例如：")]),s._v(" "),t("ul",[t("li",[s._v("*.html 匹配任意名称，扩展名为.html的文件。")]),s._v(" "),t("li",[s._v("/folder1/"),t("em",[s._v("/")]),s._v(".java 匹配在folder1目录下的任意两级目录下的.java文件。")]),s._v(" "),t("li",[s._v("/folder2/**/*.jsp 匹配在folder2目录下任意目录深度的.jsp文件。")]),s._v(" "),t("li",[s._v("/{type}/{id}.html 匹配任意文件名为{id}.html，在任意命名的{type}目录下的文件。")])]),s._v(" "),t("p",[s._v("注意：Ant 风格的路径模式语法中的特殊字符需要转义，如：")]),s._v(" "),t("ul",[t("li",[s._v("要匹配文件路径中的星号，则需要转义为\\*。")]),s._v(" "),t("li",[s._v("要匹配文件路径中的问号，则需要转义为\\?。")])]),s._v(" "),t("h3",{attrs:{id:"_2-模式切换"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_2-模式切换"}},[s._v("#")]),s._v(" 2. 模式切换")]),s._v(" "),t("blockquote",[t("p",[t("strong",[s._v("AntPathMatcher")]),s._v(" 与 "),t("code",[s._v("PathPatternParser")])]),s._v(" "),t("ul",[t("li",[t("p",[s._v("PathPatternParser 在 jmh 基准测试下，有 6~8 倍吞吐量提升，降低 30%~40%空间分配率")])]),s._v(" "),t("li",[t("p",[s._v("PathPatternParser 兼容 AntPathMatcher语法，并支持更多类型的路径模式")])]),s._v(" "),t("li",[t("p",[s._v('PathPatternParser  "'),t("em",[t("strong",[s._v("*")]),s._v('" '),t("strong",[s._v("多段匹配")]),s._v("的支持")]),t("em",[s._v("仅允许在模式末尾使用")]),s._v("*")])])])]),s._v(" "),t("div",{staticClass:"language-java line-numbers-mode"},[t("pre",{pre:!0,attrs:{class:"language-java"}},[t("code",[t("span",{pre:!0,attrs:{class:"token annotation punctuation"}},[s._v("@GetMapping")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"/a*/b?/{p1:[a-f]+}"')]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("//例如http://localhost:8080/aaaaa/ba/aabbccff")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("public")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("String")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("hello")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("HttpServletRequest")]),s._v(" request"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v(" \n                    "),t("span",{pre:!0,attrs:{class:"token annotation punctuation"}},[s._v("@PathVariable")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"p1"')]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("String")]),s._v(" path"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("{")]),s._v("\n\n    log"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("info")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"路径变量p1： {}"')]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v(" path"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(";")]),s._v("\n    "),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("//获取请求路径")]),s._v("\n    "),t("span",{pre:!0,attrs:{class:"token class-name"}},[s._v("String")]),s._v(" uri "),t("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),s._v(" request"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("getRequestURI")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(";")]),s._v("\n    "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("return")]),s._v(" uri"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(";")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("}")]),s._v("\n")])]),s._v(" "),t("div",{staticClass:"line-numbers-wrapper"},[t("span",{staticClass:"line-number"},[s._v("1")]),t("br"),t("span",{staticClass:"line-number"},[s._v("2")]),t("br"),t("span",{staticClass:"line-number"},[s._v("3")]),t("br"),t("span",{staticClass:"line-number"},[s._v("4")]),t("br"),t("span",{staticClass:"line-number"},[s._v("5")]),t("br"),t("span",{staticClass:"line-number"},[s._v("6")]),t("br"),t("span",{staticClass:"line-number"},[s._v("7")]),t("br"),t("span",{staticClass:"line-number"},[s._v("8")]),t("br"),t("span",{staticClass:"line-number"},[s._v("9")]),t("br")])]),t("p",[s._v("总结：")]),s._v(" "),t("ul",[t("li",[s._v("使用默认的路径匹配规则，是由 PathPatternParser  提供的")]),s._v(" "),t("li",[s._v("如果路径中间需要有 **，替换成ant风格路径")])]),s._v(" "),t("div",{staticClass:"language-properties line-numbers-mode"},[t("pre",{pre:!0,attrs:{class:"language-properties"}},[t("code",[t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("# 改变路径匹配策略：")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("# ant_path_matcher 老版策略；")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("# path_pattern_parser 新版策略；")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token key attr-name"}},[s._v("spring.mvc.pathmatch.matching-strategy")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token value attr-value"}},[s._v("ant_path_matcher")]),s._v("\n")])]),s._v(" "),t("div",{staticClass:"line-numbers-wrapper"},[t("span",{staticClass:"line-number"},[s._v("1")]),t("br"),t("span",{staticClass:"line-number"},[s._v("2")]),t("br"),t("span",{staticClass:"line-number"},[s._v("3")]),t("br"),t("span",{staticClass:"line-number"},[s._v("4")]),t("br")])])])}),[],!1,null,null,null);t.default=e.exports}}]);