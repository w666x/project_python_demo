# -*- coding: utf-8 -*-
# +
"""打package包，一般配置"""

from setuptools import setup, find_packages, find_namespace_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
   name="pkg_demo",
   version="1.0.0",
   description="A Python package demo",
   long_description=long_description,
   long_description_content_type="text/markdown",
   url="https://github.com",
   author="hello",
   classifiers=[
       "Natural Language :: English",
       "Natural Language :: Chinese(Simplified)",
       "Natural Language :: Chinese(Traditional)",
       "Programming Language :: Python :: 3.8",
       "Programming Language :: Python :: 3.9",
       "Programming Language :: Python :: 3 :: Only",
       "Programming Language :: python :: 3",
   ],
   keywords="ocr, pkgs",
   project_urls={
       "Source": "https://github.com/pypa/sampleproject/",
   },
   package_dir={"tt_eval": "pkgs_demo.tt_eval"},
   # packages=find_packages(),  # exclude=("*test*",)
   # packages=find_packages(include=["sample", "sample.*"]),
   packages=find_packages(where="tt_eval"),
   py_modules=["test"],
   python_requires="~=3.8",
   install_requires=["pandas", "numpy"],
   package_data={
       "": ["*.ini"],
   },
   data_files=[
       (
           "tt_eval/datas",
           ["datas/test1.csv"],
       ),
       ("tt_eval/tests", ["tests/test_echo.py"]),
   ],
   entry_points={
       "console_scripts": [
           "run_main=pkgs_demo.tt_eval:test_main",
       ],
   },
)
