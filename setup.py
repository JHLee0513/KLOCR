from io import open
from setuptools import setup,find_packages

requirements = []
dep_links = []
with open('requirements.txt', encoding="utf-8-sig") as f:
    deps = f.readlines()
    for dep in deps:
        if dep.startswith("--"):
            dep_links.append(dep)
        else:
            requirements.append(dep)

def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README

setup(
    name='kloser',
    package_dir={'kloser': 'src'},
    packages=['kloser'] + ['kloser.' + p for p in find_packages(where='src')],
    include_package_data=True,
    zip_safe=False,
    version='0.0.1',
    install_requires=requirements,
    dependency_links = dep_links,
    license='Apache License 2.0',
    description='Korean Language OCR',
    long_description=readme(),
    long_description_content_type="text/markdown",
    author='JoonHo Lee',
)
