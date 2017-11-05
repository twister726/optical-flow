Download:
- http://lear.inrialpes.fr/src/deepmatching/code/deepmatching_1.2.2.zip
- http://pascal.inrialpes.fr/data2/deepmatching/files/DeepFlow_release2.0.tar.gz

Extract them in lib

Type this in both directories:
```
make
make python
```
(You might need swig)

Final tree should look like
```
$ tree lib
lib
├── DeepFlow_release2.0
│   ├── deepflow2
│   ├── deepflow2.py
│   ├── .....
└── deepmatching_1.2.2_c++
    ├── deepmatching
    ├── deepmatching.py
    ├── .....
```
