from setuptools import find_packages, setup

setup(
    name="yeahml",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # a|b|c|rc|alpha|beta|pre|preview
    # major.minor[.patch[.sub]], but I'm not ready to call it 0.1 yet
    version="0.0.1a1",
)
