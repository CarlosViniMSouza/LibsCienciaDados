"""
Installing Beautiful Soup
If you’re using a recent version of Debian or Ubuntu Linux, you can install Beautiful Soup with the system package
manager:

$ apt-get install python-bs4 (for Python 2)

$ apt-get install python3-bs4 (for Python 3)

Beautiful Soup 4 is published through PyPi, so if you can’t install it with the system packager, you can install
it with easy_install or pip. The package name is beautifulsoup4, and the same package works on Python 2 and 
Python 3. Make sure you use the right version of pip or easy_install for your Python version (these may be named 
pip3 and easy_install3 respectively if you’re using Python 3).

$ easy_install beautifulsoup4

$ pip install beautifulsoup4

(The BeautifulSoup package is probably not what you want. That’s the previous major release, Beautiful Soup 3. 
Lots of software uses BS3, so it’s still available, but if you’re writing new code you should install 
beautifulsoup4.)

If you don’t have easy_install or pip installed, you can download the Beautiful Soup 4 source tarball and install
it with setup.py.

$ python setup.py install

If all else fails, the license for Beautiful Soup allows you to package the entire library with your application.
 You can download the tarball, copy its bs4 directory into your application’s codebase, and use Beautiful Soup 
 without installing it at all.

I use Python 2.7 and Python 3.2 to develop Beautiful Soup, but it should work with other recent versions.
"""