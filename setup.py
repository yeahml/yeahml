from setuptools import find_packages, setup
from datetime import datetime

setup(
    name="yeahml-nightly",
    version=datetime.now().strftime("0.0.1a1.dev%Y%m%d"),
    author="Jack Burdick",
    author_email='jackbburdick@gmail.com',
    description="""YeahML is a prototype library for building, training, and analyzing neural
networks using primarily configuration files""",
    long_description="TODO: Fill in",
    packages=find_packages(where='src/'),
    package_dir={"yeahml": "src/yeahml"},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'yeahml=yeahml.cli:main'
        ]
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    test_suite='tests',
    zip_safe=True,
    license="Apache license",
    url='https://github.com/yeahml/yeahml',
    keywords='yeahml,ml,configuration,baselining',
)
