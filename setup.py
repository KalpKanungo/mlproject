from setuptools import find_packages,setup
from typing import List

hypen="-e ."
def get_requirements(file_path:str)->List[str]:
    "this function will return the list of req"
    requirements=[]
    with open("requirements.txt") as file:
        requirements=file.readlines()
        requirements=[reqs.replace("\n","") for reqs in requirements]
        if hypen in requirements:
            requirements.remove(hypen)
    return requirements


setup(
name="ml-project",
version="0.0.1",
author="Kalp",
author_email="kanungokalpinox@gmail.com",
packages=find_packages(),
install_requires=get_requirements("requirements.txt")
)
