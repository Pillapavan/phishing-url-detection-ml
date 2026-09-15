'''
The setup.py file is an essential part of packaging and 
distributing Python projects. It is used by setuptools 
(or distutils in older Python versions) to define the configuration 
of your project, such as its metadata, dependencies, and more
'''


from setuptools import find_packages,setup
from typing import List


def get_requirements() ->List[str]:
    """
        This function will return list of requirements
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            lines = file.readlines()

            for line in lines:
                requirement= line.strip()
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

        

setup(
    name="phishing-detection",
    version="0.0.1",
    author="Pavan",
    author_email="pillapavan90909@gmail.com",
    description="End-to-end machine learning project for phishing URL detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements()
)