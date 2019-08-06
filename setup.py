from setuptools import setup, find_packages

setup(name='qdboard',
      version="0.0.1",
      include_package_data=True,
      install_requires=[
          'numpy',
          'untangle',
          'Flask',
          'Jinja2',
          'python-interface',
          'stopit'
      ],
      packages=find_packages()
)
