from setuptools import find_packages, setup

def get_requirements(file_path):
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    requirements = [req.strip() for req in requirements if req.strip()]
    
    return requirements



setup(
    name='vrushal',
    version='0.0.1',
    author='Vrushal More',
    author_email='vrushal.more.work@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt'),
    description='A package for various utilities and tools' 
)
