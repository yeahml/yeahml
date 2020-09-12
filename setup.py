from setuptools import find_packages, setup

setup(
    name="yeahml",
    version="0.0.1a1",
    author="Jack Burdick",
    author_email='jburdick@gmail.com',
    description="Configuration Markup for ML Libraries",
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
