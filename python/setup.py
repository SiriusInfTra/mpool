import shutil
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import os
import glob

class PostInstallCommand:
    """Post-installation for installation mode."""

    def run(self, install_lib):
        self.py_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mpool'))
        self.build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
        self.target_dir = os.path.join(install_lib, 'mpool')
        self._install_lib = install_lib
        self.copy_python_file()
        self.copy_shared_library()
        self.generate_stub()
    
    def copy_from_to(self, src, dst):
        if os.path.exists(dst):
            os.remove(dst)
            
        os.symlink(src, dst)
        print(f'make symbolic link: {src} -> {dst}.')

    def copy_python_file(self):
        for file in glob.glob(os.path.join(self.py_dir, '*.py')):
            self.copy_from_to(file, os.path.join(self.target_dir, os.path.basename(file)))

    def copy_shared_library(self):
        for dependency_library in ['pages_pool', 'allocator', 'python']:
            for so_path in glob.glob(os.path.join(self.build_dir, dependency_library, '*.so')):
                self.copy_from_to(so_path, os.path.join(self.target_dir, os.path.basename(so_path)))
    
    def generate_stub(self):
        env = os.environ.copy()
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = os.path.abspath(self._install_lib)
        else:
            env['PYTHONPATH'] += ':' + os.path.abspath(self._install_lib)
        subprocess.check_call(['pybind11-stubgen', 'mpool', '-o', self._install_lib, '--ignore-all-errors'], env=env)
    
class PostInstallCommandInstall(install, PostInstallCommand):

    def run(self):
        install.run(self)
        PostInstallCommand.run(self, self.install_lib)
    
class PostInstallCommandDevelop(develop, PostInstallCommand):
    '''
    FIXME current `pip install -e .` will not trigger this command.
    '''

    def run(self):
        develop.run(self)
        PostInstallCommand.run(self, os.path.dirname(__file__))

setup(
    name='mpool',
    version='1.0.0',
    packages=find_packages(),
    cmdclass={
        'install': PostInstallCommandInstall,
        'develop': PostInstallCommandDevelop,
    },
    zip_safe=False,
)
