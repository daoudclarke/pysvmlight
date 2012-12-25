pysvmlight
==========

Bismillahi-r-Rahmani-r-Rahim
In the Name of God, the Merciful, the Compassionate

This is a wrapper for the svmlight library. It allows you to specify
an unbiased hyperplane. It also allows you to access the learnt
hyperplane after training.

See the svmlight website (http://svmlight.joachims.org/) for full
details. 

Example of use:

```python
>>> f = DocumentFactory()
>>> docs = [f.new(x.split()) for x in [
...         "this is a nice long document",
...         "this is another nice long document",
...         "this is rather a short document",
...         "a horrible document",
...         "another horrible document"]]
>>> l = Learner()
>>> model = l.learn(docs, [1, 1, 1, -1, -1])
>>> judgments = [model.classify(d) for d in docs]
>>> print model.plane, model.bias
```

Building from source
====================

Building requires Cython to be installed. Type
```
$ python setup.py build
$ python setup.py install
```
in the root directory to build and install.


License
=======

The original svmlight code is included in the lib directory for ease
of building. This code is Copyright (c) 2002 Thorsten Joachims - All
rights reserved.

The cython code in the src directory is released under the MIT
license:

Copyright (c) 2012 Daoud Clarke

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
