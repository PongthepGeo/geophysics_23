from setuptools import setup, find_packages

setup(
    name='seismic',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'deepwave'
    ],
    url='https://github.com/PongthepGeo/geophysics_23/',
    author='PongthepGeo',
    author_email='pongtep_tong@hotmail.com',
    description='A seismic library',
)


# pip install git+https://github.com/PongthepGeo/geophysics_23.git#subdirectory=codes/seismic

