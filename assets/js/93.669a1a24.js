(window.webpackJsonp=window.webpackJsonp||[]).push([[93],{410:function(t,s,a){"use strict";a.r(s);var n=a(8),e=Object(n.a)({},(function(){var t=this,s=t._self._c;return s("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[s("blockquote",[s("p",[t._v("SpringBoot 整合 "),s("code",[t._v("Spring")]),t._v("、"),s("code",[t._v("SpringMVC")]),t._v("、"),s("code",[t._v("MyBatis")]),t._v(" 进行"),s("strong",[t._v("数据访问场景")]),t._v("开发")])]),t._v(" "),s("h2",{attrs:{id:"_1-创建ssm整合项目"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_1-创建ssm整合项目"}},[t._v("#")]),t._v(" 1. 创建SSM整合项目")]),t._v(" "),s("div",{staticClass:"language-xml line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-xml"}},[s("code",[s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("\x3c!-- https://mvnrepository.com/artifact/org.mybatis.spring.boot/mybatis-spring-boot-starter --\x3e")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("<")]),t._v("dependency")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("\n    "),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("<")]),t._v("groupId")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("org.mybatis.spring.boot"),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("</")]),t._v("groupId")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("\n    "),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("<")]),t._v("artifactId")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("mybatis-spring-boot-starter"),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("</")]),t._v("artifactId")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("\n    "),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("<")]),t._v("version")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("3.0.1"),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("</")]),t._v("version")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("</")]),t._v("dependency")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("<")]),t._v("dependency")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("\n    "),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("<")]),t._v("groupId")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("mysql"),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("</")]),t._v("groupId")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("\n    "),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("<")]),t._v("artifactId")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("mysql-connector-java"),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("</")]),t._v("artifactId")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("\n    "),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("<")]),t._v("scope")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("runtime"),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("</")]),t._v("scope")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token tag"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("</")]),t._v("dependency")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(">")])]),t._v("\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br"),s("span",{staticClass:"line-number"},[t._v("6")]),s("br"),s("span",{staticClass:"line-number"},[t._v("7")]),s("br"),s("span",{staticClass:"line-number"},[t._v("8")]),s("br"),s("span",{staticClass:"line-number"},[t._v("9")]),s("br"),s("span",{staticClass:"line-number"},[t._v("10")]),s("br"),s("span",{staticClass:"line-number"},[t._v("11")]),s("br")])]),s("h2",{attrs:{id:"_2-配置数据源"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_2-配置数据源"}},[t._v("#")]),t._v(" 2. 配置数据源")]),t._v(" "),s("div",{staticClass:"language-properties line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-properties"}},[s("code",[s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.url")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("jdbc:mysql://192.168.200.100:3306/demo")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.driver-class-name")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("com.mysql.cj.jdbc.Driver")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.username")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("root")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.password")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("123456")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.type")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("com.zaxxer.hikari.HikariDataSource")]),t._v("\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br")])]),s("p",[t._v("安装MyBatisX 插件，帮我们生成Mapper接口的xml文件即可")]),t._v(" "),s("h2",{attrs:{id:"_3-配置mybatis"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_3-配置mybatis"}},[t._v("#")]),t._v(" 3. 配置MyBatis")]),t._v(" "),s("div",{staticClass:"language-properties line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-properties"}},[s("code",[s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#指定mapper映射文件位置")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("mybatis.mapper-locations")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("classpath:/mapper/*.xml")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#参数项调整")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("mybatis.configuration.map-underscore-to-camel-case")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("true")]),t._v("\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br")])]),s("h2",{attrs:{id:"_4-crud编写"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_4-crud编写"}},[t._v("#")]),t._v(" 4. CRUD编写")]),t._v(" "),s("ul",[s("li",[t._v("编写Bean")]),t._v(" "),s("li",[t._v("编写Mapper")]),t._v(" "),s("li",[t._v("使用"),s("code",[t._v("mybatisx")]),t._v("插件，快速生成MapperXML")]),t._v(" "),s("li",[t._v("测试CRUD")])]),t._v(" "),s("h2",{attrs:{id:"_5-自动配置原理"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_5-自动配置原理"}},[t._v("#")]),t._v(" 5. 自动配置原理")]),t._v(" "),s("p",[s("strong",[t._v("SSM整合总结：")])]),t._v(" "),s("ol",[s("li",[s("p",[s("strong",[t._v("导入")]),t._v(" "),s("code",[t._v("mybatis-spring-boot-starter")])])]),t._v(" "),s("li",[s("p",[t._v("配置"),s("strong",[t._v("数据源")]),t._v("信息")])]),t._v(" "),s("li",[s("p",[t._v("配置mybatis的**"),s("code",[t._v("mapper接口扫描")]),s("strong",[t._v("与")]),s("code",[t._v("xml映射文件扫描")]),t._v("**")])]),t._v(" "),s("li",[s("p",[t._v("编写bean，mapper，生成xml，编写sql 进行crud。"),s("strong",[t._v("事务等操作依然和Spring中用法一样")])])]),t._v(" "),s("li",[s("p",[t._v("效果：")]),t._v(" "),s("ol",[s("li",[s("p",[t._v("所有sql写在xml中")])]),t._v(" "),s("li",[s("p",[t._v("所有"),s("code",[t._v("mybatis配置")]),t._v("写在"),s("code",[t._v("application.properties")]),t._v("下面")])])])])]),t._v(" "),s("ul",[s("li",[s("p",[s("code",[t._v("jdbc场景的自动配置")]),t._v("：")]),t._v(" "),s("ul",[s("li",[s("code",[t._v("mybatis-spring-boot-starter")]),t._v("导入 "),s("code",[t._v("spring-boot-starter-jdbc")]),t._v("，jdbc是操作数据库的场景")])]),t._v(" "),s("ul",[s("li",[s("p",[s("code",[t._v("Jdbc")]),t._v("场景的几个自动配置")]),t._v(" "),s("ul",[s("li",[s("p",[t._v("org.springframework.boot.autoconfigure.jdbc."),s("strong",[t._v("DataSourceAutoConfiguration")])]),t._v(" "),s("ul",[s("li",[s("p",[s("strong",[t._v("数据源的自动配置")])])]),t._v(" "),s("li",[s("p",[t._v("所有和数据源有关的配置都绑定在"),s("code",[t._v("DataSourceProperties")])])]),t._v(" "),s("li",[s("p",[t._v("默认使用 "),s("code",[t._v("HikariDataSource")])])])])]),t._v(" "),s("li",[s("p",[t._v("org.springframework.boot.autoconfigure.jdbc."),s("strong",[t._v("JdbcTemplateAutoConfiguration")])]),t._v(" "),s("ul",[s("li",[t._v("给容器中放了"),s("code",[t._v("JdbcTemplate")]),t._v("操作数据库（小工具）")])])]),t._v(" "),s("li",[s("p",[t._v("org.springframework.boot.autoconfigure.jdbc."),s("strong",[t._v("JndiDataSourceAutoConfiguration")])])]),t._v(" "),s("li",[s("p",[t._v("org.springframework.boot.autoconfigure.jdbc."),s("strong",[t._v("XADataSourceAutoConfiguration")])]),t._v(" "),s("ul",[s("li",[s("strong",[t._v("基于XA二阶提交协议的分布式事务数据源")])])])]),t._v(" "),s("li",[s("p",[t._v("org.springframework.boot.autoconfigure.jdbc."),s("strong",[t._v("DataSourceTransactionManagerAutoConfiguration")])]),t._v(" "),s("ul",[s("li",[t._v("放了一个事务管理器，"),s("strong",[t._v("支持事务")])])])])])])]),t._v(" "),s("ul",[s("li",[s("strong",[t._v("具有的底层能力：数据源、")]),s("code",[t._v("JdbcTemplate")]),t._v("、"),s("strong",[t._v("事务")])])])]),t._v(" "),s("li",[s("p",[s("code",[t._v("MyBatisAutoConfiguration")]),t._v("：配置了MyBatis的整合流程")]),t._v(" "),s("ul",[s("li",[s("code",[t._v("mybatis-spring-boot-starter")]),t._v("导入 "),s("code",[t._v("mybatis-spring-boot-autoconfigure（mybatis的自动配置包）")]),t._v("，")])]),t._v(" "),s("ul",[s("li",[s("p",[t._v("默认加载两个自动配置类，在mybatis的autoconfigure包下的"),s("code",[t._v("/META-INF/spirng/org.springframework.boot.autoconfigure.AutoConfiguration.imports")]),t._v("文件中：")]),t._v(" "),s("ul",[s("li",[s("p",[t._v("org.mybatis.spring.boot.autoconfigure.MybatisLanguageDriverAutoConfiguration")])]),t._v(" "),s("li",[s("p",[t._v("org.mybatis.spring.boot.autoconfigure."),s("strong",[t._v("MybatisAutoConfiguration")])]),t._v(" "),s("ul",[s("li",[s("p",[s("strong",[t._v("必须在数据源配置好之后才配置")])])]),t._v(" "),s("li",[s("p",[t._v("给容器中放"),s("code",[t._v("SqlSessionFactory")]),t._v("组件。创建和数据库的一次会话（CRUD操作）")])]),t._v(" "),s("li",[s("p",[t._v("给容器中放"),s("code",[t._v("SqlSessionTemplate")]),t._v("组件。操作数据库")])])])])])])]),t._v(" "),s("ul",[s("li",[s("strong",[t._v("MyBatis的所有配置绑定在")]),s("code",[t._v("MybatisProperties")])])]),t._v(" "),s("ul",[s("li",[t._v("每个"),s("strong",[t._v("Mapper接口")]),t._v("的"),s("strong",[t._v("代理对象")]),t._v("是怎么创建放到容器中。详见**@MapperScan**原理：\n"),s("ul",[s("li",[t._v("利用"),s("code",[t._v("@Import(MapperScannerRegistrar.class)")]),t._v("批量给容器中注册组件。解析指定的包路径里面的每一个类，为每一个Mapper接口类，创建Bean定义信息，注册到容器中。")])])])])])]),t._v(" "),s("blockquote",[s("p",[t._v("如何分析哪个场景导入以后，开启了哪些自动配置类。")]),t._v(" "),s("p",[t._v("找："),s("code",[t._v("classpath:/META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports")]),t._v("文件中配置的所有值，就是要开启的自动配置类，但是每个类可能有条件注解，基于条件注解判断哪个自动配置类生效了。")])]),t._v(" "),s("h2",{attrs:{id:"_6-快速定位生效的配置"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_6-快速定位生效的配置"}},[t._v("#")]),t._v(" 6. 快速定位生效的配置")]),t._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("#开启调试模式，详细打印开启了哪些自动配置\ndebug=true\n# Positive（生效的自动配置）  Negative（不生效的自动配置）\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br")])]),s("h2",{attrs:{id:"_7-扩展-整合其他数据源"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_7-扩展-整合其他数据源"}},[t._v("#")]),t._v(" 7. 扩展：整合其他数据源")]),t._v(" "),s("h3",{attrs:{id:"_1-druid-数据源"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_1-druid-数据源"}},[t._v("#")]),t._v(" 1. Druid 数据源")]),t._v(" "),s("p",[t._v("暂不支持 "),s("code",[t._v("SpringBoot3")])]),t._v(" "),s("ul",[s("li",[t._v("导入"),s("code",[t._v("druid-starter")])]),t._v(" "),s("li",[t._v("写配置")]),t._v(" "),s("li",[t._v("分析自动配置了哪些东西，怎么用")])]),t._v(" "),s("p",[t._v("Druid官网：https://github.com/alibaba/druid")]),t._v(" "),s("div",{staticClass:"language-properties line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-properties"}},[s("code",[s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#数据源基本配置")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.url")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("jdbc:mysql://192.168.200.100:3306/demo")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.driver-class-name")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("com.mysql.cj.jdbc.Driver")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.username")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("root")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.password")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("123456")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.type")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("com.alibaba.druid.pool.DruidDataSource")]),t._v("\n\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 配置StatFilter监控")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.druid.filter.stat.enabled")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("true")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.druid.filter.stat.db-type")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("mysql")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.druid.filter.stat.log-slow-sql")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("true")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.druid.filter.stat.slow-sql-millis")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("2000")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 配置WallFilter防火墙")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.druid.filter.wall.enabled")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("true")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.druid.filter.wall.db-type")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("mysql")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.druid.filter.wall.config.delete-allow")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("false")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.druid.filter.wall.config.drop-table-allow")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("false")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 配置监控页，内置监控页面的首页是 /druid/index.html")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.druid.stat-view-servlet.enabled")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("true")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.druid.stat-view-servlet.login-username")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("admin")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.druid.stat-view-servlet.login-password")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("admin")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token key attr-name"}},[t._v("spring.datasource.druid.stat-view-servlet.allow")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token value attr-value"}},[t._v("*")]),t._v("\n\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 其他 Filter 配置不再演示")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 目前为以下 Filter 提供了配置支持，请参考文档或者根据IDE提示（spring.datasource.druid.filter.*）进行配置。")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# StatFilter")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# WallFilter")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# ConfigFilter")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# EncodingConvertFilter")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# Slf4jLogFilter")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# Log4jFilter")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# Log4j2Filter")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# CommonsLogFilter")]),t._v("\n\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br"),s("span",{staticClass:"line-number"},[t._v("6")]),s("br"),s("span",{staticClass:"line-number"},[t._v("7")]),s("br"),s("span",{staticClass:"line-number"},[t._v("8")]),s("br"),s("span",{staticClass:"line-number"},[t._v("9")]),s("br"),s("span",{staticClass:"line-number"},[t._v("10")]),s("br"),s("span",{staticClass:"line-number"},[t._v("11")]),s("br"),s("span",{staticClass:"line-number"},[t._v("12")]),s("br"),s("span",{staticClass:"line-number"},[t._v("13")]),s("br"),s("span",{staticClass:"line-number"},[t._v("14")]),s("br"),s("span",{staticClass:"line-number"},[t._v("15")]),s("br"),s("span",{staticClass:"line-number"},[t._v("16")]),s("br"),s("span",{staticClass:"line-number"},[t._v("17")]),s("br"),s("span",{staticClass:"line-number"},[t._v("18")]),s("br"),s("span",{staticClass:"line-number"},[t._v("19")]),s("br"),s("span",{staticClass:"line-number"},[t._v("20")]),s("br"),s("span",{staticClass:"line-number"},[t._v("21")]),s("br"),s("span",{staticClass:"line-number"},[t._v("22")]),s("br"),s("span",{staticClass:"line-number"},[t._v("23")]),s("br"),s("span",{staticClass:"line-number"},[t._v("24")]),s("br"),s("span",{staticClass:"line-number"},[t._v("25")]),s("br"),s("span",{staticClass:"line-number"},[t._v("26")]),s("br"),s("span",{staticClass:"line-number"},[t._v("27")]),s("br"),s("span",{staticClass:"line-number"},[t._v("28")]),s("br"),s("span",{staticClass:"line-number"},[t._v("29")]),s("br"),s("span",{staticClass:"line-number"},[t._v("30")]),s("br"),s("span",{staticClass:"line-number"},[t._v("31")]),s("br"),s("span",{staticClass:"line-number"},[t._v("32")]),s("br"),s("span",{staticClass:"line-number"},[t._v("33")]),s("br"),s("span",{staticClass:"line-number"},[t._v("34")]),s("br")])]),s("h2",{attrs:{id:"附录-示例数据库"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#附录-示例数据库"}},[t._v("#")]),t._v(" 附录：示例数据库")]),t._v(" "),s("div",{staticClass:"language-sql line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-sql"}},[s("code",[s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("CREATE")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("TABLE")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token identifier"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("`")]),t._v("t_user"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("`")])]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("\n    "),s("span",{pre:!0,attrs:{class:"token identifier"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("`")]),t._v("id"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("`")])]),t._v("         "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("BIGINT")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("20")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("   "),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("NOT")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("NULL")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("AUTO_INCREMENT")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("COMMENT")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token string"}},[t._v("'编号'")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n    "),s("span",{pre:!0,attrs:{class:"token identifier"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("`")]),t._v("login_name"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("`")])]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("VARCHAR")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("200")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("NULL")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("DEFAULT")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("NULL")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("COMMENT")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token string"}},[t._v("'用户名称'")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("COLLATE")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token string"}},[t._v("'utf8_general_ci'")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n    "),s("span",{pre:!0,attrs:{class:"token identifier"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("`")]),t._v("nick_name"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("`")])]),t._v("  "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("VARCHAR")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("200")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("NULL")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("DEFAULT")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("NULL")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("COMMENT")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token string"}},[t._v("'用户昵称'")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("COLLATE")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token string"}},[t._v("'utf8_general_ci'")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n    "),s("span",{pre:!0,attrs:{class:"token identifier"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("`")]),t._v("passwd"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("`")])]),t._v("     "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("VARCHAR")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("200")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("NULL")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("DEFAULT")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("NULL")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("COMMENT")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token string"}},[t._v("'用户密码'")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("COLLATE")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token string"}},[t._v("'utf8_general_ci'")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n    "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("PRIMARY")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("KEY")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),s("span",{pre:!0,attrs:{class:"token identifier"}},[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("`")]),t._v("id"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("`")])]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("insert")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("into")]),t._v(" t_user"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("login_name"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" nick_name"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" passwd"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("VALUES")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),s("span",{pre:!0,attrs:{class:"token string"}},[t._v("'zhangsan'")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),s("span",{pre:!0,attrs:{class:"token string"}},[t._v("'张三'")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),s("span",{pre:!0,attrs:{class:"token string"}},[t._v("'123456'")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")])])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br"),s("span",{staticClass:"line-number"},[t._v("6")]),s("br"),s("span",{staticClass:"line-number"},[t._v("7")]),s("br"),s("span",{staticClass:"line-number"},[t._v("8")]),s("br")])])])}),[],!1,null,null,null);s.default=e.exports}}]);