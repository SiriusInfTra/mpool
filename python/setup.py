from __future__ import annotations
import subprocess
from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info
import os
import glob
import sys

class PostInstallCommand:
    """Post-installation for installation mode."""

    def run(self, build_lib: str, install_lib: str):
        self.binary_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
        self.install_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), install_lib))
        self.build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), build_lib))
        print('self.install_dir = ', self.install_dir)
        print('self.build_dir = ', self.build_dir)
        self.copy_python_file()
        self.copy_shared_library()
        self.generate_stub()
    
    def copy_from_to(self, src, dst):
        if os.path.abspath(src) == os.path.abspath(dst):
            return
        if os.path.exists(dst):
            os.remove(dst)
            
        os.symlink(src, dst)
        print(f'make symbolic link: {src} -> {dst}.')

    def copy_python_file(self):
        for file in glob.glob(os.path.join(self.build_dir, 'mpool', '*.py')):
            self.copy_from_to(file, os.path.join(self.install_dir, 'mpool', os.path.basename(file)))

    def copy_shared_library(self):
        for dependency_library in ['pages_pool', 'allocator', 'python']:
            for so_path in glob.glob(os.path.join(self.binary_dir, dependency_library, '*.so')):
                self.copy_from_to(so_path, os.path.join(self.install_dir, 'mpool', os.path.basename(so_path)))
    
    def generate_stub(self):
        env = os.environ.copy()
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = os.path.abspath(self.install_dir)
        else:
            env['PYTHONPATH'] = os.path.abspath(self.install_dir) + ':' + env['PYTHONPATH']
        subprocess.check_call(['pybind11-stubgen', 'mpool', '-o', self.install_dir, '--ignore-all-errors'], env=env)

class PostInstallCommandEggInfo(egg_info, PostInstallCommand):
    
    def run(self):
        egg_info.run(self)
        build_lib = os.path.dirname(__file__)
        install_lib = os.path.dirname(__file__)
        if 'egg_info' not in sys.argv or '-' in os.path.basename(build_lib):
            return
        PostInstallCommand.run(self, build_lib, install_lib)

setup(
    name='mpool',
    version='1.0.0',
    packages=find_packages(),
    cmdclass={
        'egg_info': PostInstallCommandEggInfo,
    },
    zip_safe=False,
)
