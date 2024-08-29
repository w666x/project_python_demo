# -*- coding: utf-8 -*-
# +
"""打package包，命名空间版，多个相关包打一起"""

from setuptools import setup, find_packages, find_namespace_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
   name="pkgs_demo",
   version="1.0.0",
   description="A Python namespace package demo",
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
   keywords="namespace, Display, pkgs",
   project_urls={
       "Source": "https://github.com/pypa/sampleproject/",
   },
   packages=find_namespace_packages(include=["pkgs_demo.*"]),
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
