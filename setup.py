from setuptools import setup, find_packages


install_requires=[
   'matplotlib>=3.2.2',
   'numpy>=1.18.5,<1.24.0',
   'opencv-python>=4.1.1',
   'Pillow>=7.1.2',
   'PyYAML>=5.3.1',
   'requests>=2.23.0',
   'scipy>=1.4.1',
   'torch>=1.7.0,!=1.12.0',
   'torchvision>=0.8.1,!=0.13.0',
   'tqdm>=4.41.0',
   'protobuf<4.21.3',
   'tensorboard>=2.4.1',
   'pandas>=1.1.4',
   'seaborn>=0.11.0',
]

setup(
    name="tactus - yolov7",
    version="0.0.1",
    description="Yolov7 library for tactus",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
)
