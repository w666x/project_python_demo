# project_python_demo


## 打package


### 开始打包

<!-- #region -->
- 1. 开始打包

```sh
# cd切到setup.py对应的文件下
sh build.sh # 打包，发布到pypi一条龙
```
<!-- #endregion -->

<!-- #region -->
- 2. 验证库是否OK

```sh
# 1. 终端直接执行，entry_points对应命令
run_main

# 2. python环境import对应Python库
from pkg_demo import tt_eval
tt_eval.test_main()
```
<!-- #endregion -->

### setup.py介绍

- 常用功能如下所示，详细功能，可见下列文件
    - [打常用的包](setup.py)
    - [打命名空间包](setup_env.py)


| 是否常用 | 配置项                  | 功能描述                                       | demo
|:-|:-|:-|:-
| 是 | `name`               | 包的名称，通常是项目的唯一标识。                      |
| 是 | `version`            | 包的版本号，遵循PEP 440版本格式。                   |
| 是 | `author`             | 包的作者姓名。                                   |
| 是 | `author_email`       | 作者的电子邮件地址。                              |
|  | `maintainer`         | 维护者的姓名（如果与作者不同）。                     |
|  | `maintainer_email`   | 维护者的电子邮件地址。                             |
| 是 | `url`                | 项目的主页或源代码仓库的URL。                      | git地址即可
| 是 | `license`            | 包使用的许可证。                                  |
| 是 | `description`        | 包的简短描述。                                   |
| 是 | `long_description`   | 包的长描述                | 通常从README文件中获取
| 是 | `keywords`           | 与包相关的关键词列表。                             |
| 是 | `classifiers`        | 按照PyPI的分类标准对包进行分类。                     | 
|  | `platforms`          | 指定包兼容的平台。                                |
| 是 | `packages`           | 要包含在分发中的包和子包列表。                      | `packages=find_packages(include=["sample", "sample.*"])`
| 是 | `py_modules`        | 要包含的单个Python模块列表, **正常情况是不在package中的文件**。                       | `py_modules=["test"]`
|  | `scripts`            | 包含在分发中的可执行脚本列表。                      |
|  | `ext_modules`       | 定义的扩展模块列表，用于编译C/C++扩展。             |
|  | `cmdclass`           | 用于自定义命令的字典。                             |
| 是 | `data_files`         | 包含非Python数据文件的列表。                        | 数据放在package外面的文件夹下面
| 是 | `package_dir`       | 指定包目录的映射。                                | `package_dir={"tt_eval": "pkgs_demo.tt_eval"}`
| 是 | `package_data`      | 指定要包含在包中的非代码文件。                       | 比如配置文件，在package目录下
| 是 | `include_package_data` | 指示是否从MANIFEST.in文件中包含数据文件。            |
|  | `zip_safe`           | 指示分发是否安全地作为zip归档分发。                   |
| 是 | `install_requires`  | 项目运行所需的依赖列表。                           | `install_requires=["pandas", "numpy"]`
|  | `extras_require`    | 指定额外的依赖组，用于特定功能或可选组件。           |
| 是 | `entry_points`      | 定义包的入口点，用于插件和应用程序集成。            | 把对应命令写到环境变量中去，可直接执行
|  | `test_suite`        | 指定用于运行测试的测试套件。                        |
|  | `tests_require`     | 测试所需的额外依赖。                                |
| 是 | `python_requires`   | 指定兼容的Python版本范围。                          | `python_requires="~=3.8"`


## 环境说明

<!-- #region -->
- 1. 项目结构

```sh
├── build.sh        # 打包的shell命令
├── code_review.md    # code review帮助文档，忽略
├── datas          # 项目数据或资源文件目录，包含配置文件、数据集等
├── docs           # 文档目录，包含项目文档的源文件，如Sphinx文档
├── LICENSE.txt      # 项目的许可证文件，指定了项目的法律条款和使用限制
├── MANIFEST.in      # 指定非代码文件（如文档、数据文件等）哪些需要包含在源分发源中
├── pics
├── pkgs_demo        # 源代码目录，包含项目的主要代码
├── README.md
├── requirements.txt   # 包含项目运行所需的依赖列表，通常由`pip`管理
├── setup.cfg        # 包含`setup.py`的配置选项，可以用于覆盖或添加额外的配置
├── setup_env.py      # 项目的设置和配置文件，和下面的重复，打命名空间包
├── setup.py        # 项目的设置和配置文件，用于定义包的元数据、依赖关系、入口点等
├── tests          # 包含项目测试代码的目录
└── version.py       # 有时项目会包含一个版本文件，用于存储当前版本号
```
<!-- #endregion -->

<!-- #region -->
- 2. 环境说明-cv代码

```sh
python==3.8.10
ubuntu==20.04.1
GPU==RTX A6000
cuda version==11.4
```

```sh
torch==1.12.0.dev20220327+cu113
torchkeras==3.9.9
```
<!-- #endregion -->
