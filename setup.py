"""Package Setup script for tfx_x."""
import os

from setuptools import find_packages
from setuptools import setup


def select_constraint(default, nightly=None, git_master=None):
  """Select dependency constraint based on TFX_DEPENDENCY_SELECTOR env var."""
  selector = os.environ.get('TFX_DEPENDENCY_SELECTOR')
  if selector == 'UNCONSTRAINED':
    return ''
  elif selector == 'NIGHTLY' and nightly is not None:
    return nightly
  elif selector == 'GIT_MASTER' and git_master is not None:
    return git_master
  else:
    return default


# Get version from version module.
with open('tfx_x/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']


def _make_required_install_packages():
  # Make sure to sync the versions of common dependencies (absl-py, numpy,
  # six, and protobuf) with TF.
  return [
    # LINT.IfChange
    'apache-beam[gcp]>=2.27,<3',
    'click>=7,<8',
    'google-api-python-client>=1.7.8,<2',
    'grpcio>=1.28.1,<2',
    'numpy>=1.16,<1.20',
    'pyarrow>=1,<3',
    'pyyaml>=3.12,<6',
    'tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<3',
    'tfx-bsl' + select_constraint(
      default='>=0.27,<0.28',
      nightly='>=0.28.0.dev',
      git_master='@git+https://github.com/tensorflow/tfx-bsl@master'),
  ]


# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setup(
  name='tfx_x',
  version=__version__,
  author='Sebastien Soudan',
  author_email='sebastien.soudan@gmail.com',
  license='Apache 2.0',
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
  ],
  namespace_packages=[],
  install_requires=_make_required_install_packages(),
  python_requires='>=3.6,<4',
  packages=find_packages(),
  include_package_data=True,
  description='A library to extend TFX',
  long_description=_LONG_DESCRIPTION,
  long_description_content_type='text/markdown',
  keywords='tensorflow transform tfx',
  url='https://github.com/ssoudan/tfx_x',
  download_url='https://github.com/ssoudan/tfx_x/tags',
  requires=[])
