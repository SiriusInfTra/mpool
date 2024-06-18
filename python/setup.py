import logging
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
import pybind11_stubgen as stubgen
import os
import shutil
import glob

class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        self.py_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mpool'))
        self.build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
        self.target_dir = os.path.join(self.install_lib, 'mpool')
        self.copy_python_file()
        self.copy_shared_library()
        self.generate_stub()
    
    def copy_python_file(self):
        for file in glob.glob(os.path.join(self.py_dir, '*.py')):
            shutil.copy(file, self.target_dir)

    def copy_shared_library(self):
        for dependency_library in ['pages_pool', 'allocator', 'python']:
            for so_path in glob.glob(os.path.join(self.build_dir, dependency_library, '*.so')):
                shutil.copy(so_path, self.target_dir)
    
    def generate_stub(self):
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = os.path.abspath(self.install_lib)
        else:
            env['PYTHONPATH'] += ':' + os.path.abspath(self.install_lib)
        subprocess.check_call(['pybind11-stubgen', 'mpool', '-o', self.install_lib], env=env)

setup(
    name='mpool',
    version='1.0.0',
    packages=find_packages(),
    cmdclass={
        'install': PostInstallCommand,
    },
    zip_safe=False,
)
