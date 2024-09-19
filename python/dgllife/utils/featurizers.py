# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Node and edge featurization for molecular graphs.
# pylint: disable= no-member, arguments-differ, invalid-name

import itertools
import os.path as osp

from collections import defaultdict
from functools import partial

import numpy as np
import torch
import dgl.backend as F

try:
    from rdkit import Chem, RDConfig
    from rdkit.Chem import AllChem, ChemicalFeatures
except ImportError:
    pass
#Python 模块的 __all__ 列表。__all__ 是一个特殊的变量，用于定义当使用 from module import * 语法时，模块中应该暴露给用户的对象或函数的名称。也就是说，__all__ 列表中列出的名字是模块的公开接口
__all__ = ['one_hot_encoding',
           'atom_type_one_hot',
           'atomic_number_one_hot',
           'atomic_number',
           'atom_degree_one_hot',
           'atom_degree',
           'atom_total_degree_one_hot',
           'atom_total_degree',
           'atom_explicit_valence_one_hot',
           'atom_explicit_valence',
           'atom_implicit_valence_one_hot',
           'atom_implicit_valence',
           'atom_hybridization_one_hot',
           'atom_total_num_H_one_hot',
           'atom_total_num_H',
           'atom_formal_charge_one_hot',
           'atom_formal_charge',
           'atom_num_radical_electrons_one_hot',
           'atom_num_radical_electrons',
           'atom_is_aromatic_one_hot',
           'atom_is_aromatic',
           'atom_is_in_ring_one_hot',
           'atom_is_in_ring',
           'atom_chiral_tag_one_hot',
           'atom_chirality_type_one_hot',
           'atom_mass',
           'atom_is_chiral_center',
           'ConcatFeaturizer',
           'BaseAtomFeaturizer',
           'CanonicalAtomFeaturizer',
           'WeaveAtomFeaturizer',
           'PretrainAtomFeaturizer',
           'AttentiveFPAtomFeaturizer',
           'PAGTNAtomFeaturizer',
           'bond_type_one_hot',
           'bond_is_conjugated_one_hot',
           'bond_is_conjugated',
           'bond_is_in_ring_one_hot',
           'bond_is_in_ring',
           'bond_stereo_one_hot',
           'bond_direction_one_hot',
           'BaseBondFeaturizer',
           'CanonicalBondFeaturizer',
           'WeaveEdgeFeaturizer',
           'PretrainBondFeaturizer',
           'AttentiveFPBondFeaturizer',
           'PAGTNEdgeFeaturizer']

def one_hot_encoding(x, allowable_set, encode_unknown=False):
    """One-hot encoding.

    Parameters
    ----------
    x
        Value to encode.
    allowable_set : list
        The elements of the allowable_set should be of the
        same type as x.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element.  #这个是指用于映射那些不再list中的元素，若为True 就自己在末尾增添一个维度，那这样，就不能保证每个节点的维度一致了呀

    Returns
    -------
    list
        List of boolean values where at most one value is True.
        The list is of length ``len(allowable_set)`` if ``encode_unknown=False``
        and ``len(allowable_set) + 1`` otherwise.

    Examples
    --------
    >>> from dgllife.utils import one_hot_encoding
    >>> one_hot_encoding('C', ['C', 'O'])
    [True, False]
    >>> one_hot_encoding('S', ['C', 'O'])
    [False, False]
    >>> one_hot_encoding('S', ['C', 'O'], encode_unknown=True)
    [False, False, True]
    """
    if encode_unknown and (allowable_set[-1] is not None):  #这种方式保证维度一致
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: x == s, allowable_set))

#################################################################
# Atom featurization
#################################################################

def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False): #看来一般他是不管超出规定集的东西的
    """One hot encoding for the type of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of str
        Atom types to consider. Default: ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``,
        ``Cl``, ``Br``, ``Mg``, ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``,
        ``K``, ``Tl``, ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
        ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``, ``Cr``,
        ``Pt``, ``Hg``, ``Pb``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atomic_number
    atomic_number_one_hot
    """
    if allowable_set is None:
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)
#atom.GetSymbol() 是 RDKit（一个化学信息学工具包）中 Atom 对象的一个方法，用于获取原子的化学符号。

def atomic_number_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the atomic number of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atomic numbers to consider. Default: ``1`` - ``100``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atomic_number
    atom_type_one_hot
    """
    if allowable_set is None:
        allowable_set = list(range(1, 101))
    return one_hot_encoding(atom.GetAtomicNum(), allowable_set, encode_unknown)
    #GetAtomicNum() 是 RDKit 中 Atom 类的一个方法，用于获取原子的原子序数

def atomic_number(atom):
    """Get the atomic number for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
       List containing one int only.

    See Also
    --------
    atomic_number_one_hot
    atom_type_one_hot
    """
    return [atom.GetAtomicNum()] #atom.GetAtomicNum() 是 RDKit（一个化学信息学工具包）中的一个方法，用于获取一个原子的原子序数（atomic number）

def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the degree of an atom.

    Note that the result will be different depending on whether the Hs are
    explicitly modeled in the graph.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom degrees to consider. Default: ``0`` - ``10``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_degree
    atom_total_degree
    atom_total_degree_one_hot
    """
    if allowable_set is None:
        allowable_set = list(range(11))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)

def atom_degree(atom):
    """Get the degree of an atom.

    Note that the result will be different depending on whether the Hs are
    explicitly modeled in the graph.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_degree_one_hot
    atom_total_degree
    atom_total_degree_one_hot
    """
    return [atom.GetDegree()]

def atom_total_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the degree of an atom including Hs.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list
        Total degrees to consider. Default: ``0`` - ``5``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    See Also
    --------
    one_hot_encoding
    atom_degree
    atom_degree_one_hot
    atom_total_degree
    """
    if allowable_set is None:
        allowable_set = list(range(6))
    return one_hot_encoding(atom.GetTotalDegree(), allowable_set, encode_unknown)
#atom.GetTotalDegree() 是 RDKit 中 Atom 类的一个方法，用于获取一个原子的总连接度（total degree）。连接度是指一个原子通过化学键直接连接到其他原子的数量。
def atom_total_degree(atom):
    """The degree of an atom including Hs.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_total_degree_one_hot
    atom_degree
    atom_degree_one_hot
    """
    return [atom.GetTotalDegree()]

def atom_explicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the explicit valence of an aotm.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom explicit valences to consider. Default: ``1`` - ``6``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_explicit_valence
    """
    if allowable_set is None:
        allowable_set = list(range(1, 7))
    return one_hot_encoding(atom.GetExplicitValence(), allowable_set, encode_unknown)
#显式价态更强调通过化学键连接的原子数，而连接度则强调与原子直接相连的原子数量。
def atom_explicit_valence(atom):
    """Get the explicit valence of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_explicit_valence_one_hot
    """
    return [atom.GetExplicitValence()]  #atom.GetExplicitValence() 方法用于获取原子的显式价态

def atom_implicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the implicit valence of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom implicit valences to consider. Default: ``0`` - ``6``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    atom_implicit_valence
    """
    if allowable_set is None:
        allowable_set = list(range(7))
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)
#atom.GetImplicitValence() 是 RDKit 中 Atom 类的一个方法，用于获取原子的隐式价态（implicit valence）。
#隐式价态指的是一个原子通过隐含的氢原子或其他未明确显示的连接所填补的价电子空缺。
def atom_implicit_valence(atom):
    """Get the implicit valence of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Reurns
    ------
    list
        List containing one int only.

    See Also
    --------
    atom_implicit_valence_one_hot
    """
    return [atom.GetImplicitValence()]

# pylint: disable=I1101
def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the hybridization of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of rdkit.Chem.rdchem.HybridizationType
        Atom hybridizations to consider. Default: ``Chem.rdchem.HybridizationType.SP``,
        ``Chem.rdchem.HybridizationType.SP2``, ``Chem.rdchem.HybridizationType.SP3``,
        ``Chem.rdchem.HybridizationType.SP3D``, ``Chem.rdchem.HybridizationType.SP3D2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.HybridizationType.SP,               #线性杂化（sp）。
                         Chem.rdchem.HybridizationType.SP2,              #平面三角形杂化（sp²）
                         Chem.rdchem.HybridizationType.SP3,              #四面体杂化（sp³）
                         Chem.rdchem.HybridizationType.SP3D,             #五面体杂化（sp³d），虽然在有机分子中不常见
                         Chem.rdchem.HybridizationType.SP3D2]            #八面体杂化（sp³d²），也在有机化学中较少见
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)
#用于获取原子的杂化状态（hybridization）。杂化状态描述了原子中价电子轨道的混合方式，从而决定了原子如何形成化学键和其几何形状。
def atom_total_num_H_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the total number of Hs of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Total number of Hs to consider. Default: ``0`` - ``4``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_total_num_H
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)
#获取一个原子的总氢原子数（包括隐式氢）。

def atom_total_num_H(atom):
    """Get the total number of Hs of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_total_num_H_one_hot
    """
    return [atom.GetTotalNumHs()]

def atom_formal_charge_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the formal charge of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Formal charges to consider. Default: ``-2`` - ``2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_formal_charge
    """
    if allowable_set is None:
        allowable_set = list(range(-2, 3))
    return one_hot_encoding(atom.GetFormalCharge(), allowable_set, encode_unknown)
#获取一个原子的形式电荷（formal charge）。形式电荷是用于描述原子在化学结构中实际带有的电荷量的一种方式。
def atom_formal_charge(atom):
    """Get formal charge for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_formal_charge_one_hot
    """
    return [atom.GetFormalCharge()]

def atom_partial_charge(atom):
    """Get Gasteiger partial charge for an atom.

    For using this function, you must have called ``AllChem.ComputeGasteigerCharges(mol)``
    to compute Gasteiger charges.
#在 RDKit 中，Gasteiger 充电（Gasteiger charges）是计算分子中每个原子的电荷分布的一种方法。
#这种方法基于分子的结构和原子的电子环境，提供了原子的电荷信息，帮助理解分子中的电荷分布情况。
    Occasionally, we can get nan or infinity Gasteiger charges, in which case we will set
    the result to be 0.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one float only.
    """
    gasteiger_charge = atom.GetProp('_GasteigerCharge')
    if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
        gasteiger_charge = 0
    return [float(gasteiger_charge)]
#atom 对象中获取 _GasteigerCharge 属性的值。这个属性是在调用 AllChem.ComputeGasteigerCharges(mol) 后计算并存储在原子对象中的，用于表示 Gasteiger 电荷。


def atom_num_radical_electrons_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the number of radical electrons of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Number of radical electrons to consider. Default: ``0`` - ``4``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_num_radical_electrons
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetNumRadicalElectrons(), allowable_set, encode_unknown)
#用于获取原子上的自由基电子数（radical electrons）。自由基电子是指原子上不成对的电子，这些电子参与了自由基的形成，并对分子的反应性有重要影响。
def atom_num_radical_electrons(atom):
    """Get the number of radical electrons for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_num_radical_electrons_one_hot
    """
    return [atom.GetNumRadicalElectrons()]

def atom_is_aromatic_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the atom is aromatic.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_is_aromatic
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.GetIsAromatic(), allowable_set, encode_unknown)
#用于检查一个原子是否属于芳香环系统。芳香环系统是指具有芳香性的环状结构
def atom_is_aromatic(atom):
    """Get whether the atom is aromatic.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    atom_is_aromatic_one_hot
    """
    return [atom.GetIsAromatic()]

def atom_is_in_ring_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the atom is in ring.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_is_in_ring
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.IsInRing(), allowable_set, encode_unknown)
#原子是否在环上
def atom_is_in_ring(atom):
    """Get whether the atom is in ring.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    atom_is_in_ring_one_hot
    """
    return [atom.IsInRing()]

def atom_chiral_tag_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the chiral tag of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of rdkit.Chem.rdchem.ChiralType
        Chiral tags to consider. Default: ``rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_OTHER``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    one_hot_encoding
    atom_chirality_type_one_hot
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,              #未指定的立体化学标记。通常表示该原子不是立体中心或立体化学配置未被定义。
                         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,           #表示原子是一个四面体立体化学中心，配置为反时针（Counterclockwise，通常表示 S 配置）。
                         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,          #表示原子是一个四面体立体化学中心，配置为顺时针（Clockwise，通常表示 R 配置）。
                         Chem.rdchem.ChiralType.CHI_OTHER]                    #用于处理不符合常规四面体立体化学标记（如 CHI_TETRAHEDRAL_CW 或 CHI_TETRAHEDRAL_CCW）的特殊情况。
    return one_hot_encoding(atom.GetChiralTag(), allowable_set, encode_unknown)
#atom.GetChiralTag() 获取原子的立体化学标记。
def atom_chirality_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the chirality type of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of str
        Chirality types to consider. Default: ``R``, ``S``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    one_hot_encoding
    atom_chiral_tag_one_hot
    """
    if not atom.HasProp('_CIPCode'): #这行代码检查原子是否具有 _CIPCode 属性。 _CIPCode 是用于表示立体化学配置的属性，通常是 R 或 S。
        return [False, False]

    if allowable_set is None:
        allowable_set = ['R', 'S']
    return one_hot_encoding(atom.GetProp('_CIPCode'), allowable_set, encode_unknown)
#获取属性值：通过 atom.GetProp('_CIPCode') 获取原子的 _CIPCode 属性值。
#调用 one_hot_encoding 函数：将获取的 _CIPCode 属性值、allowable_set 和 encode_unknown 参数传递给 one_hot_encoding 函数，返回该原子的 one-hot 编码。


def atom_mass(atom, coef=0.01):
    """Get the mass of an atom and scale it.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    coef : float
        The mass will be multiplied by ``coef``.

    Returns
    -------
    list
        List containing one float only.
    """
    return [atom.GetMass() * coef]
#atom.GetMass() * coef 这一表达式通常是为了调整或加权原子的质量，
def atom_is_chiral_center(atom):
    """Get whether the atom is chiral center #要确定一个原子是否是手性中心（chiral center），需要检查该原子的立体化学属性。

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.
    """
    return [atom.HasProp('_ChiralityPossible')]
#HasProp: 这是一个方法，用于检查原子是否具有特定的属性。它接受一个属性名称作为参数，并返回一个布尔值，指示该属性是否存在。
#'_ChiralityPossible': 这是一个特定的属性名称。它的作用是标记原子是否可能是一个手性中心。

class ConcatFeaturizer(object):
    """Concatenate the evaluation results of multiple functions as a single feature.
    #将多个函数的特征结果连接（concatenate）成为一个单一的特征（feature）。

    Parameters
    ----------
    func_list : list
        List of functions for computing molecular descriptors from objects of a same
        particular data type, e.g. ``rdkit.Chem.rdchem.Atom``. Each function is of signature
        ``func(data_type) -> list of float or bool or int``. The resulting order of
        the features will follow that of the functions in the list.
#计算分子描述符的函数列表，这些函数作用于相同特定数据类型的对象，例如 rdkit.Chem.rdchem.Atom。
#每个函数的签名为 func(data_type) -> list of float or bool or int。结果特征的顺序将遵循函数列表中的顺序。
    Examples
    --------

    Setup for demo.

    >>> from dgllife.utils import ConcatFeaturizer
    >>> from rdkit import Chem
    >>> smi = 'CCO'
    >>> mol = Chem.MolFromSmiles(smi)

    Concatenate multiple atom descriptors as a single node feature.

    >>> from dgllife.utils import atom_degree, atomic_number, BaseAtomFeaturizer
    >>> # Construct a featurizer for featurizing one atom a time
    >>> atom_concat_featurizer = ConcatFeaturizer([atom_degree, atomic_number])
    >>> # Construct a featurizer for featurizing all atoms in a molecule
    >>> mol_atom_featurizer = BaseAtomFeaturizer({'h': atom_concat_featurizer})
    >>> mol_atom_featurizer(mol)
    {'h': tensor([[1., 6.],
                  [2., 6.],
                  [1., 8.]])}

    Conctenate multiple bond descriptors as a single edge feature.

    >>> from dgllife.utils import bond_type_one_hot, bond_is_in_ring, BaseBondFeaturizer
    >>> # Construct a featurizer for featurizing one bond a time
    >>> bond_concat_featurizer = ConcatFeaturizer([bond_type_one_hot, bond_is_in_ring])
    >>> # Construct a featurizer for featurizing all bonds in a molecule
    >>> mol_bond_featurizer = BaseBondFeaturizer({'h': bond_concat_featurizer})
    >>> mol_bond_featurizer(mol)
    {'h': tensor([[1., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0.]])}
    """
#列表中的每个函数都需要接受 x 作为输入，并返回一个包含浮点数、布尔值或整数的列表。这些函数可能用于提取不同的分子描述符或特征。
#使用示例：
#如果 self.func_list 包含函数 func1（计算分子的原子数）和 func2（计算分子的键数），那么：
#func1(x) 可能返回 [6]（代表分子中有 6 个原子）。
#func2(x) 可能返回 [5]（代表分子中有 5 个键）。
#通过 [func(x) for func in self.func_list]，你得到 [[6], [5]]。
#itertools.chain.from_iterable 会将这些列表合并成 [6, 5]。
#最终，list(...) 会将其转换为 [6, 5] 作为函数的返回结果。

    def __init__(self, func_list):
        self.func_list = func_list

    def __call__(self, x):
        """Featurize the input data.

        Parameters
        ----------
        x :
            Data to featurize.

        Returns
        #x 是你要传递给函数列表中每个函数的数据。在上面的示例中，x 是一个 RDKit 的分子对象。它可能是任何你希望传递给这些函数的对象，具体取决于函数的预期输入类型
        -------
        list
            List of feature values, which can be of type bool, float or int.
        """
        return list(itertools.chain.from_iterable(
            [func(x) for func in self.func_list]))
#itertools.chain.from_iterable 是 itertools 模块中的一个函数，用于将一个可迭代对象（这里是一个包含列表的列表）扁平化成一个单一的迭代器。

class BaseAtomFeaturizer(object):
    """An abstract class for atom featurizers. #一个用于原子特征化的抽象类

    Loop over all atoms in a molecule and featurize them with the ``featurizer_funcs``.
#遍历分子中的所有原子，并使用 featurizer_funcs 对它们进行特征化
    **We assume the resulting DGLGraph will not contain any virtual nodes and a node i in the
    graph corresponds to exactly atom i in the molecule.**
#我们假设生成的 DGLGraph 不会包含虚拟节点，并且图中的节点 i 正好对应分子中的第 i 个原子
    Parameters
    ----------
    featurizer_funcs : dict
        Mapping feature name to the featurization function.
        Each function is of signature ``func(rdkit.Chem.rdchem.Atom) -> list or 1D numpy array``.
    feat_sizes : dict
        Mapping feature name to the size of the corresponding feature. If None, they will be
        computed when needed. Default: None.
#featurizer_funcs : dict
#特征名称与特征化函数的映射。
#每个函数的签名为 func(rdkit.Chem.rdchem.Atom) -> list 或 1D numpy 数组。
#feat_sizes : dict
#特征名称与相应特征大小的映射。如果为 None，将在需要时计算它们。默认值：None。
    Examples
    --------

    >>> from dgllife.utils import BaseAtomFeaturizer, atom_mass, atom_degree_one_hot
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = BaseAtomFeaturizer({'mass': atom_mass, 'degree': atom_degree_one_hot})
    >>> atom_featurizer(mol)
    {'mass': tensor([[0.1201],
                     [0.1201],
                     [0.1600]]),
     'degree': tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])}
    >>> # Get feature size for atom mass
    >>> print(atom_featurizer.feat_size('mass'))
    1
    >>> # Get feature size for atom degree
    >>> print(atom_featurizer.feat_size('degree'))
    11

    See Also
    --------
    CanonicalAtomFeaturizer
    WeaveAtomFeaturizer
    PretrainAtomFeaturizer
    AttentiveFPAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, featurizer_funcs, feat_sizes=None):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes

    def feat_size(self, feat_name=None):
        """Get the feature size for ``feat_name``.

        When there is only one feature, users do not need to provide ``feat_name``.

        Parameters
        ----------
        feat_name : str
            Feature for query.

        Returns
        -------
        int
            Feature size for the feature with name ``feat_name``. Default to None.
        """
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        if feat_name not in self._feat_sizes:
            atom = Chem.MolFromSmiles('C').GetAtomWithIdx(0)
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](atom))

        return self._feat_sizes[feat_name]

#确保 feat_name 是有效的，并且是 self.featurizer_funcs 中定义的特征名称。
#如果 feat_name 没有指定，但特征化函数字典中只有一个函数，则自动选择该函数的名称。
#如果 feat_name 对应的特征大小尚未计算，则使用示例原子计算特征的大小，并缓存结果。最终返回特征的大小。
    def __call__(self, mol):
        """Featurize all atoms in a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            For each function in self.featurizer_funcs with the key ``k``, store the computed
            feature under the key ``k``. Each feature is a tensor of dtype float32 and shape
            (N, M), where N is the number of atoms in the molecule.
        """
        num_atoms = mol.GetNumAtoms() #原子总数
        atom_features = defaultdict(list)

        # Compute features for each atom
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)  #atom = mol.GetAtomWithIdx(i) 这行代码从一个 RDKit 分子对象 mol 中根据索引 i 获取对应的原子对象。这是处理分子结构时常用的方法，可以用于提取和分析分子中的每个原子信息。
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))
#对于每个特征名称和特征化函数，使用 feat_func 函数计算原子的特征。将计算结果附加到 atom_features[feat_name] 中。
#atom_features 是一个字典，其键是特征名称，值是一个列表，用于存储该特征的所有原子特征。假设 atom_features 已经被初始化为包含每个特征名称的空列表。
        # Stack the features and convert them to float arrays  #代码的目的是将存储在 atom_features 字典中的原子特征转换为一个处理后的特征字典 processed_features
        processed_features = dict()
        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list)
            #arrays: 需要堆叠的数组序列。所有的数组必须具有相同的形状，axis: 堆叠的轴。默认是 0，即沿着第一个轴堆叠。
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))
            
#F.zerocopy_from_numpy 是 DGL 库中 dgl.backend 模块的一个函数，用于将 NumPy 数组转换为 DGL 张量（tensor），并确保转换过程中不进行数据复制。这种方式可以提高效率，特别是在数据量较大时。


        return processed_features

class CanonicalAtomFeaturizer(BaseAtomFeaturizer):
    """A default featurizer for atoms.

    The atom features include:

    * **One hot encoding of the atom type**. The supported atom types include
      ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``, ``Cl``, ``Br``, ``Mg``,
      ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``, ``K``, ``Tl``,
      ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
      ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``,
      ``Cr``, ``Pt``, ``Hg``, ``Pb``.
    * **One hot encoding of the atom degree**. The supported possibilities
      include ``0 - 10``.
    * **One hot encoding of the number of implicit Hs on the atom**. The supported
      possibilities include ``0 - 6``.
    * **Formal charge of the atom**.
    * **Number of radical electrons of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include ``0 - 4``.

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import CanonicalAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                      1., 0.],
                     [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
                      0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
                      0., 0.]])}

    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size('feat'))
    74

    See Also
    --------
    BaseAtomFeaturizer
    WeaveAtomFeaturizer
    PretrainAtomFeaturizer
    AttentiveFPAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atom_data_field='h'):
        super(CanonicalAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [atom_type_one_hot,
                 atom_degree_one_hot,
                 atom_implicit_valence_one_hot,
                 atom_formal_charge,
                 atom_num_radical_electrons,
                 atom_hybridization_one_hot,
                 atom_is_aromatic,
                 atom_total_num_H_one_hot]
            )})

class WeaveAtomFeaturizer(object):
    """Atom featurizer in Weave.
#在“分子图卷积：超越指纹（Molecular Graph Convolutions: Moving Beyond Fingerprints）”一文中，
#涉及到的 原子特征化 是指将分子中的原子属性和特征转换为可以用于机器学习模型的数值格式的过程
    The atom featurization performed in `Molecular Graph Convolutions: Moving Beyond Fingerprints
    <https://arxiv.org/abs/1603.00856>`__, which considers:

    * atom types
    * chirality
    * formal charge
    * partial charge
    * aromatic atom
    * hybridization
    * hydrogen bond donor
    * hydrogen bond acceptor
    * the number of rings the atom belongs to for ring size between 3 and 8

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.
    atom_types : list of str or None
        Atom types to consider for one-hot encoding. If None, we will use a default
        choice of ``'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'``.
    chiral_types : list of Chem.rdchem.ChiralType or None
        Atom chirality to consider for one-hot encoding. If None, we will use a default
        choice of ``Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW``.
    hybridization_types : list of Chem.rdchem.HybridizationType or None
        Atom hybridization types to consider for one-hot encoding. If None, we will use a
        default choice of ``Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3``.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import WeaveAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = WeaveAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0418,  0.0000,
                       0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000],
                     [ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0402,  0.0000,
                       0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000],
                     [ 0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.3967,  0.0000,
                       0.0000,  0.0000,  1.0000,  1.0000,  1.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000]])}
    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size())
    27

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    PretrainAtomFeaturizer
    AttentiveFPAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atom_data_field='h', atom_types=None, chiral_types=None,
                 hybridization_types=None):
        super(WeaveAtomFeaturizer, self).__init__()

        self._atom_data_field = atom_data_field

        if atom_types is None:
            atom_types = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        self._atom_types = atom_types

        if chiral_types is None:
            chiral_types = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, #手性中心的顺时针（CW）
                            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW] #逆时针（CCW）排列决定了其具体的立体化学性质。
        self._chiral_types = chiral_types

        if hybridization_types is None:
            hybridization_types = [Chem.rdchem.HybridizationType.SP,
                                   Chem.rdchem.HybridizationType.SP2,
                                   Chem.rdchem.HybridizationType.SP3]
        self._hybridization_types = hybridization_types

        self._featurizer = ConcatFeaturizer([
            partial(atom_type_one_hot, allowable_set=atom_types, encode_unknown=True),
            partial(atom_chiral_tag_one_hot, allowable_set=chiral_types),
            atom_formal_charge, atom_partial_charge, atom_is_aromatic,
            partial(atom_hybridization_one_hot, allowable_set=hybridization_types)
        ])

        fdef_name = osp.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        self._mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
#fdef_name = osp.join(RDConfig.RDDataDir, "BaseFeatures.fdef")：
#osp.join 是 os.path.join 的缩写，它用于生成一个文件路径。
#RDConfig.RDDataDir 是 RDKit 配置中的一个变量，指向 RDKit 数据文件的目录。
#"BaseFeatures.fdef" 是一个定义了化学特征的文件名。它包含了如何识别和定义分子中的特征（例如，官能团、环等）

#self._mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)：

#ChemicalFeatures 是 RDKit 中一个模块，提供了处理化学特征的功能。
#BuildFeatureFactory 是 ChemicalFeatures 模块中的一个方法，它接受一个特征定义文件的路径（如 fdef_name），并返回一个“特征工厂”（Feature Factory）对象。
#特征工厂对象可以用来从分子中提取特征，这些特征是基于 BaseFeatures.fdef 文件中的定义。特征提取是将化学信息转化为可用于机器学习或其他计算分析的格式的过程。
    def feat_size(self):
        """Get the feature size.

        Returns
        -------
        int
            Feature size.
        """
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self._atom_data_field]
        
#这里的 self(mol) 表示你在某个类中定义了一个 __call__ 方法，使得这个类的实例可以像函数一样被调用。self(mol) 会处理分子 mol 并返回某些结果。
#[self._atom_data_field] 表示从上述结果中提取特定的字段。self._atom_data_field 是一个类属性，指示要从结果中提取的数据字段。通常，它会是一个字符串，指定了你感兴趣的特征数据字段。
#feats 变量保存了从分子 mol 中提取的特征数据，特征数据是基于 self._atom_data_field 指定的字段。
        return feats.shape[-1]

    def get_donor_acceptor_info(self, mol_feats):
        """Bookkeep whether an atom is donor/acceptor for hydrogen bonds.

        Parameters
        ----------
        mol_feats : tuple of rdkit.Chem.rdMolChemicalFeatures.MolChemicalFeature
            Features for molecules.

        Returns
        -------
        is_donor : dict
            Mapping atom ids to binary values indicating whether atoms
            are donors for hydrogen bonds
        is_acceptor : dict
            Mapping atom ids to binary values indicating whether atoms
            are acceptors for hydrogen bonds
        """
        is_donor = defaultdict(bool)
        is_acceptor = defaultdict(bool)
        # Get hydrogen bond donor/acceptor information
        for feats in mol_feats:
            if feats.GetFamily() == 'Donor':
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_donor[u] = True
            elif feats.GetFamily() == 'Acceptor':
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_acceptor[u] = True

        return is_donor, is_acceptor

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping atom_data_field as specified in the input argument to the atom
            features, which is a float32 tensor of shape (N, M), N is the number of
            atoms and M is the feature size.
        """
        atom_features = []

        AllChem.ComputeGasteigerCharges(mol)
        num_atoms = mol.GetNumAtoms()

        # Get information for donor and acceptor
        mol_feats = self._mol_featurizer.GetFeaturesForMol(mol)
        is_donor, is_acceptor = self.get_donor_acceptor_info(mol_feats)

        # Get a symmetrized smallest set of smallest rings
        # Following the practice from Chainer Chemistry (https://github.com/chainer/
        # chainer-chemistry/blob/da2507b38f903a8ee333e487d422ba6dcec49b05/chainer_chemistry/
        # dataset/preprocessors/weavenet_preprocessor.py)
        sssr = Chem.GetSymmSSSR(mol)

        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            # Features that can be computed directly from RDKit atom instances, which is a list
            feats = self._featurizer(atom)
            # Donor/acceptor indicator
            feats.append(float(is_donor[i]))
            feats.append(float(is_acceptor[i]))
            # Count the number of rings the atom belongs to for ring size between 3 and 8
            count = [0 for _ in range(3, 9)]
            for ring in sssr:
                ring_size = len(ring)
                if i in ring and 3 <= ring_size <= 8:
                    count[ring_size - 3] += 1
            feats.extend(count)
            atom_features.append(feats)
        atom_features = np.stack(atom_features)

        return {self._atom_data_field: F.zerocopy_from_numpy(atom_features.astype(np.float32))}

class PretrainAtomFeaturizer(object):
    """AtomFeaturizer in Strategies for Pre-training Graph Neural Networks.

    The atom featurization performed in `Strategies for Pre-training Graph Neural Networks
    <https://arxiv.org/abs/1905.12265>`__, which considers:

    * atomic number
    * chirality

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atomic_number_types : list of int or None
        Atomic number types to consider for one-hot encoding. If None, we will use a default
        choice of 1-118.
    chiral_types : list of Chem.rdchem.ChiralType or None
        Atom chirality to consider for one-hot encoding. If None, we will use a default
        choice, including ``Chem.rdchem.ChiralType.CHI_UNSPECIFIED``,
        ``Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW``,
        ``Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW``, ``Chem.rdchem.ChiralType.CHI_OTHER``.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import PretrainAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = PretrainAtomFeaturizer()
    >>> atom_featurizer(mol)
    {'atomic_number': tensor([5, 5, 7]), 'chirality_type': tensor([0, 0, 0])}

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    WeaveAtomFeaturizer
    AttentiveFPAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atomic_number_types=None, chiral_types=None):
        if atomic_number_types is None:
            atomic_number_types = list(range(1, 119))
        self._atomic_number_types = atomic_number_types

        if chiral_types is None:
            chiral_types = [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_OTHER
            ]
        self._chiral_types = chiral_types

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping 'atomic_number' and 'chirality_type' to separately an int64 tensor
            of shape (N, 1), N is the number of atoms
        """
        atom_features = []
        num_atoms = mol.GetNumAtoms()
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom_features.append([
                self._atomic_number_types.index(atom.GetAtomicNum()),
                self._chiral_types.index(atom.GetChiralTag())
            ])
        atom_features = np.stack(atom_features)
        atom_features = F.zerocopy_from_numpy(atom_features.astype(np.int64))

        return {
            'atomic_number': atom_features[:, 0],
            'chirality_type': atom_features[:, 1]
        }

class AttentiveFPAtomFeaturizer(BaseAtomFeaturizer):
    """The atom featurizer used in AttentiveFP

    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    The atom features include:

    * **One hot encoding of the atom type**. The supported atom types include
      ``B``, ``C``, ``N``, ``O``, ``F``, ``Si``, ``P``, ``S``, ``Cl``, ``As``,
      ``Se``, ``Br``, ``Te``, ``I``, ``At``, and ``other``.
    * **One hot encoding of the atom degree**. The supported possibilities
      include ``0 - 5``.
    * **Formal charge of the atom**.
    * **Number of radical electrons of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``, and ``other``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include ``0 - 4``.
    * **Whether the atom is chiral center**
    * **One hot encoding of the atom chirality type**. The supported possibilities include
      ``R``, and ``S``.

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import AttentiveFPAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                      0., 0., 0.],
                     [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                      0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                      0., 0., 0.]])}

    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size('feat'))
    39

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    WeaveAtomFeaturizer
    PretrainAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atom_data_field='h'):
        super(AttentiveFPAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [partial(atom_type_one_hot, allowable_set=[
                    'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S',
                    'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'], encode_unknown=True),
                 partial(atom_degree_one_hot, allowable_set=list(range(6))),
                 atom_formal_charge,
                 atom_num_radical_electrons,
                 partial(atom_hybridization_one_hot, encode_unknown=True),
                 atom_is_aromatic,
                 atom_total_num_H_one_hot,
                 atom_is_chiral_center,
                 atom_chirality_type_one_hot]
            )})

class PAGTNAtomFeaturizer(BaseAtomFeaturizer):
    """The atom featurizer used in PAGTN

    PAGTN is introduced in
    `Path-Augmented Graph Transformer Network. <https://arxiv.org/abs/1905.12712>`__

    The atom features include:

    * **One hot encoding of the atom type**.
    * **One hot encoding of formal charge of the atom**.
    * **One hot encoding of the atom degree**
    * **One hot encoding of explicit valence of an atom**. The supported possibilities
      include ``0 - 6``.
    * **One hot encoding of implicit valence of an atom**. The supported possibilities
      include ``0 - 5``.
    * **Whether the atom is aromatic**.

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import PAGTNAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('C')
    >>> atom_featurizer = PAGTNAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 1., 0., 0.]])}
    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size())
    94

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    PretrainAtomFeaturizer
    WeaveAtomFeaturizer
    AttentiveFPAtomFeaturizer
    """
    def __init__(self, atom_data_field='h'):
        SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
                   'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
                   'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                   'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
                   'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re',
                   'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm',
                   'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', '*', 'UNK']

        super(PAGTNAtomFeaturizer, self).__init__(
            featurizer_funcs={
                atom_data_field: ConcatFeaturizer([partial(atom_type_one_hot,
                                                           allowable_set=SYMBOLS,
                                                           encode_unknown=False),
                                                   atom_formal_charge_one_hot,
                                                   atom_degree_one_hot,
                                                   partial(atom_explicit_valence_one_hot,
                                                           allowable_set=list(range(7)),
                                                           encode_unknown=False),
                                                   partial(atom_implicit_valence_one_hot,
                                                           allowable_set=list(range(6)),
                                                           encode_unknown=False),
                                                   atom_is_aromatic])})

def bond_type_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of a bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of Chem.rdchem.BondType
        Bond types to consider. Default: ``Chem.rdchem.BondType.SINGLE``,
        ``Chem.rdchem.BondType.DOUBLE``, ``Chem.rdchem.BondType.TRIPLE``,
        ``Chem.rdchem.BondType.AROMATIC``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondType.SINGLE,
                         Chem.rdchem.BondType.DOUBLE,
                         Chem.rdchem.BondType.TRIPLE,
                         Chem.rdchem.BondType.AROMATIC]
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)

def bond_is_conjugated_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the bond is conjugated. #键是否共轭

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    bond_is_conjugated
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.GetIsConjugated(), allowable_set, encode_unknown)
#bond.GetIsConjugated() 是 RDKit 库中的一个方法，用于确定化学键是否为共轭键
def bond_is_conjugated(bond):
    """Get whether the bond is conjugated.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    bond_is_conjugated_one_hot
    """
    return [bond.GetIsConjugated()]

def bond_is_in_ring_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the bond is in a ring of any size.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    bond_is_in_ring
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.IsInRing(), allowable_set, encode_unknown)

def bond_is_in_ring(bond):
    """Get whether the bond is in a ring of any size.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    bond_is_in_ring_one_hot
    """
    return [bond.IsInRing()]

def bond_stereo_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the stereo configuration of a bond.
    
#在化学中，立体化学配置（Stereo Configuration）指的是分子中手性中心或立体异构体的三维空间排列。对于化学键，立体化学配置可以包括：
#顺式（Cis）与反式（Trans）：在一些含有双键的分子中，描述取代基的空间关系。
#R和S：用于描述手性中心的立体异构体。
#E和Z：描述双键两侧取代基的空间关系
    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of rdkit.Chem.rdchem.BondStereo
        Stereo configurations to consider. Default: ``rdkit.Chem.rdchem.BondStereo.STEREONONE``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOANY``, ``rdkit.Chem.rdchem.BondStereo.STEREOZ``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOE``, ``rdkit.Chem.rdchem.BondStereo.STEREOCIS``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOTRANS``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondStereo.STEREONONE,                #无立体化学配置
                         Chem.rdchem.BondStereo.STEREOANY,                 #任何立体配置
                         Chem.rdchem.BondStereo.STEREOZ,                   #Z 配置（在双键上，两个高优先级取代基在同一侧）
                         Chem.rdchem.BondStereo.STEREOE,                   #E 配置（在双键上，两个高优先级取代基在不同侧）
                         Chem.rdchem.BondStereo.STEREOCIS,                 #顺式配置（通常用于描述双键的取代基）
                         Chem.rdchem.BondStereo.STEREOTRANS]               #反式配置（与顺式相对）
    return one_hot_encoding(bond.GetStereo(), allowable_set, encode_unknown)
#bond.GetStereo()：获取的立体配置值。

def bond_direction_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the direction of a bond.  #键的方向性

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of Chem.rdchem.BondDir
        Bond directions to consider. Default: ``Chem.rdchem.BondDir.NONE``,
        ``Chem.rdchem.BondDir.ENDUPRIGHT``, ``Chem.rdchem.BondDir.ENDDOWNRIGHT``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondDir.NONE,                                         #没有特定的方向信息。
                         Chem.rdchem.BondDir.ENDUPRIGHT,                                   #键的方向朝上或右上。
                         Chem.rdchem.BondDir.ENDDOWNRIGHT]                                 #键的方向朝下或右下。
    return one_hot_encoding(bond.GetBondDir(), allowable_set, encode_unknown)
#bond.GetBondDir()：获取的化学键方向值。

class BaseBondFeaturizer(object):
    """An abstract class for bond featurizers.
    Loop over all bonds in a molecule and featurize them with the ``featurizer_funcs``.
    We assume the constructed ``DGLGraph`` is a bi-directed graph where the **i** th bond in the
    molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the **(2i)**-th and **(2i+1)**-th edges
    in the DGLGraph.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**
#“”"一个用于键特征化的抽象类。
#循环遍历分子中的所有键，并使用 featurizer_funcs 对它们进行特征化。
#我们假设构造的 DGLGraph 是一个双向图，其中分子中的第 i 条键，即 mol.GetBondWithIdx(i)，对应于 DGLGraph 中的第 (2i) 和 (2i+1) 条边。
#我们假设生成的 DGLGraph 将通过 :func:smiles_to_bigraph 创建，并且没有自环。
    Parameters
    ----------
    featurizer_funcs : dict
        Mapping feature name to the featurization function.
        Each function is of signature ``func(rdkit.Chem.rdchem.Bond) -> list or 1D numpy array``.
    feat_sizes : dict
        Mapping feature name to the size of the corresponding feature. If None, they will be
        computed when needed. Default: None.
    self_loop : bool
        Whether self loops will be added. Default to False. If True, it will use an additional
        column of binary values to indicate the identity of self loops in each bond feature.
        The features of the self loops will be zero except for the additional columns.

#featurizer_funcs : dict
#特征名称与特征化函数的映射。
#每个函数的签名为 func(rdkit.Chem.rdchem.Bond) -> list 或 1D numpy 数组。
#feat_sizes : dict
#特征名称与相应特征大小的映射。如果为 None，则在需要时计算。默认值：None。
#self_loop : bool
#是否会添加自环。默认为 False。如果为 True，它将使用一列二进制值来指示每个键特征中的自环标识。
#自环的特征将为零，除了额外的列以外。

    Examples
    --------

    >>> from dgllife.utils import BaseBondFeaturizer, bond_type_one_hot, bond_is_in_ring
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = BaseBondFeaturizer({'type': bond_type_one_hot, 'ring': bond_is_in_ring})
    >>> bond_featurizer(mol)
    {'type': tensor([[1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [1., 0., 0., 0.]]),
     'ring': tensor([[0.], [0.], [0.], [0.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('type')
    4
    >>> bond_featurizer.feat_size('ring')
    1

    # Featurization with self loops to add

    >>> bond_featurizer = BaseBondFeaturizer(
    ...                       {'type': bond_type_one_hot, 'ring': bond_is_in_ring},
    ...                       self_loop=True)
    >>> bond_featurizer(mol)
    {'type': tensor([[1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 1.]]),
     'ring': tensor([[0., 0.],
                     [0., 0.],
                     [0., 0.],
                     [0., 0.],
                     [0., 1.],
                     [0., 1.],
                     [0., 1.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('type')
    5
    >>> bond_featurizer.feat_size('ring')
    2

    See Also
    --------
    CanonicalBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    PAGTNEdgeFeaturizer
    """
    def __init__(self, featurizer_funcs, feat_sizes=None, self_loop=False):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes
        self._self_loop = self_loop

    def feat_size(self, feat_name=None):
        """Get the feature size for ``feat_name``.

        When there is only one feature, users do not need to provide ``feat_name``.

        Parameters
        ----------
        feat_name : str
            Feature for query.

        Returns
        -------
        int
            Feature size for the feature with name ``feat_name``. Default to None.
        """
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        mol = Chem.MolFromSmiles('CCO')
        feats = self(mol)
#self(mol):当你在一个类的方法内部使用 self(mol) 时，实际上是在调用当前类实例的 __call__ 方法。__call__ 是 Python 的特殊方法，它允许类的实例像函数一样被调用
        return feats[feat_name].shape[1]

    def __call__(self, mol):
        """Featurize all bonds in a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            For each function in self.featurizer_funcs with the key ``k``, store the computed
            feature under the key ``k``. Each feature is a tensor of dtype float32 and shape
            (N, M), where N is the number of atoms in the molecule.
        """
        num_bonds = mol.GetNumBonds()
        bond_features = defaultdict(list)
#defaultdict 是 Python 的 collections 模块中的一个类，用于创建带有默认值的字典。与普通字典不同，defaultdict 在访问不存在的键时不会抛出 KeyError 异常，而是自动创建一个默认值。
#在这里，defaultdict(list) 表示每个新的键都会被初始化为一个空列表 []。
#可是list是啥？？？
        # Compute features for each bond
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                feat = feat_func(bond)
                bond_features[feat_name].extend([feat, feat.copy()])

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in bond_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        if self._self_loop and num_bonds > 0:
            num_atoms = mol.GetNumAtoms()
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.cat([feats, torch.zeros(feats.shape[0], 1)], dim=1)
                self_loop_feats = torch.zeros(num_atoms, feats.shape[1])
                self_loop_feats[:, -1] = 1
                feats = torch.cat([feats, self_loop_feats], dim=0)
                processed_features[feat_name] = feats

        if self._self_loop and num_bonds == 0:
            num_atoms = mol.GetNumAtoms()
            toy_mol = Chem.MolFromSmiles('CO')
            processed_features = self(toy_mol)
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.zeros(num_atoms, feats.shape[1])
                feats[:, -1] = 1
                processed_features[feat_name] = feats

        return processed_features

class CanonicalBondFeaturizer(BaseBondFeaturizer):
    """A default featurizer for bonds.

    The bond features include:
    * **One hot encoding of the bond type**. The supported bond types include
      ``SINGLE``, ``DOUBLE``, ``TRIPLE``, ``AROMATIC``.
    * **Whether the bond is conjugated.**.
    * **Whether the bond is in a ring of any size.**
    * **One hot encoding of the stereo configuration of a bond**. The supported bond stereo
      configurations include ``STEREONONE``, ``STEREOANY``, ``STEREOZ``, ``STEREOE``,
      ``STEREOCIS``, ``STEREOTRANS``.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**

    Parameters
    ----------
    bond_data_field : str
        Name for storing bond features in DGLGraphs, default to ``'e'``.
    self_loop : bool
        Whether self loops will be added. Default to False. If True, it will use an additional
        column of binary values to indicate the identity of self loops. The feature of the
        self loops will be zero except for the additional column.

    Examples
    --------

    >>> from dgllife.utils import CanonicalBondFeaturizer
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat')
    >>> bond_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('feat')
    12

    # Featurization with self loops to add
    >>> bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat', self_loop=True)
    >>> bond_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('feat')
    13

    See Also
    --------
    BaseBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    PAGTNEdgeFeaturizer
    """
    def __init__(self, bond_data_field='e', self_loop=False):
        super(CanonicalBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [bond_type_one_hot,
                 bond_is_conjugated,
                 bond_is_in_ring,
                 bond_stereo_one_hot]
            )}, self_loop=self_loop)

# pylint: disable=E1102
class WeaveEdgeFeaturizer(object):
    """Edge featurizer in Weave.

    The edge featurization is introduced in `Molecular Graph Convolutions:
    Moving Beyond Fingerprints <https://arxiv.org/abs/1603.00856>`__.

    This featurization is performed for a complete graph of atoms with self loops added,
    which considers:

    * Number of bonds between each pairs of atoms
    * One-hot encoding of bond type if a bond exists between a pair of atoms
    * Whether a pair of atoms belongs to a same ring

    Parameters
    ----------
    edge_data_field : str
        Name for storing edge features in DGLGraphs, default to ``'e'``.
    max_distance : int
        Maximum number of bonds to consider between each pair of atoms.
        Default to 7.
    bond_types : list of Chem.rdchem.BondType or None
        Bond types to consider for one hot encoding. If None, we consider by
        default single, double, triple and aromatic bonds.

    Examples
    --------

    >>> from dgllife.utils import WeaveEdgeFeaturizer
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CO')
    >>> edge_featurizer = WeaveEdgeFeaturizer(edge_data_field='feat')
    >>> edge_featurizer(mol)
    {'feat': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])}
    >>> edge_featurizer.feat_size()
    12

    See Also
    --------
    BaseBondFeaturizer
    CanonicalBondFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    PAGTNEdgeFeaturizer
    """
    def __init__(self, edge_data_field='e', max_distance=7, bond_types=None):
        super(WeaveEdgeFeaturizer, self).__init__()

        self._edge_data_field = edge_data_field
        self._max_distance = max_distance
        if bond_types is None:
            bond_types = [Chem.rdchem.BondType.SINGLE,
                          Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE,
                          Chem.rdchem.BondType.AROMATIC]
        self._bond_types = bond_types

    def feat_size(self):
        """Get the feature size.

        Returns
        -------
        int
            Feature size.
        """
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self._edge_data_field]

        return feats.shape[-1]

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping self._edge_data_field to a float32 tensor of shape (N, M), where
            N is the number of atom pairs and M is the feature size.
        """
        # Part 1 based on number of bonds between each pair of atoms
        distance_matrix = torch.from_numpy(Chem.GetDistanceMatrix(mol))
        # Change shape from (V, V, 1) to (V^2, 1)
        distance_matrix = distance_matrix.float().reshape(-1, 1)
        # Elementwise compare if distance is bigger than 0, 1, ..., max_distance - 1
        distance_indicators = (distance_matrix >
                               torch.arange(0, self._max_distance).float()).float()

        # Part 2 for one hot encoding of bond type.
        num_atoms = mol.GetNumAtoms()
        bond_indicators = torch.zeros(num_atoms, num_atoms, len(self._bond_types))
        for bond in mol.GetBonds():
            bond_type_encoding = torch.tensor(
                bond_type_one_hot(bond, allowable_set=self._bond_types)).float()
            begin_atom_idx, end_atom_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_indicators[begin_atom_idx, end_atom_idx] = bond_type_encoding
            bond_indicators[end_atom_idx, begin_atom_idx] = bond_type_encoding
        # Reshape from (V, V, num_bond_types) to (V^2, num_bond_types)
        bond_indicators = bond_indicators.reshape(-1, len(self._bond_types))

        # Part 3 for whether a pair of atoms belongs to a same ring.
        sssr = Chem.GetSymmSSSR(mol)
        ring_mate_indicators = torch.zeros(num_atoms, num_atoms, 1)
        for ring in sssr:
            ring = list(ring)
            num_atoms_in_ring = len(ring)
            for i in range(num_atoms_in_ring):
                ring_mate_indicators[ring[i], torch.tensor(ring)] = 1
        ring_mate_indicators = ring_mate_indicators.reshape(-1, 1)

        return {self._edge_data_field: torch.cat([distance_indicators,
                                                  bond_indicators,
                                                  ring_mate_indicators], dim=1)}

class PretrainBondFeaturizer(object):
    """BondFeaturizer in Strategies for Pre-training Graph Neural Networks.

    The bond featurization performed in `Strategies for Pre-training Graph Neural Networks
    <https://arxiv.org/abs/1905.12265>`__, which considers:

    * bond type
    * bond direction

    Parameters
    ----------
    bond_types : list of Chem.rdchem.BondType or None
        Bond types to consider. Default to ``Chem.rdchem.BondType.SINGLE``,
        ``Chem.rdchem.BondType.DOUBLE``, ``Chem.rdchem.BondType.TRIPLE``,
        ``Chem.rdchem.BondType.AROMATIC``.
    bond_direction_types : list of Chem.rdchem.BondDir or None
        Bond directions to consider. Default to ``Chem.rdchem.BondDir.NONE``,
        ``Chem.rdchem.BondDir.ENDUPRIGHT``, ``Chem.rdchem.BondDir.ENDDOWNRIGHT``.
    self_loop : bool
        Whether self loops will be added. Default to True.

    Examples
    --------

    >>> from dgllife.utils import PretrainBondFeaturizer
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CO')
    >>> bond_featurizer = PretrainBondFeaturizer()
    >>> bond_featurizer(mol)
    {'bond_type': tensor([0, 0, 4, 4]),
     'bond_direction_type': tensor([0, 0, 0, 0])}
    """
    def __init__(self, bond_types=None, bond_direction_types=None, self_loop=True):
        if bond_types is None:
            bond_types = [
                Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
            ]
        self._bond_types = bond_types

        if bond_direction_types is None:
            bond_direction_types = [
                Chem.rdchem.BondDir.NONE,
                Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.ENDDOWNRIGHT
            ]
        self._bond_direction_types = bond_direction_types
        self._self_loop = self_loop

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping 'bond_type' and 'bond_direction_type' separately to an int64
            tensor of shape (N, 1), where N is the number of edges.
        """
        edge_features = []
        num_bonds = mol.GetNumBonds()
        if num_bonds == 0:
            assert self._self_loop, \
                'The molecule has 0 bonds and we should set self._self_loop to True.'

        # Compute features for each bond
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            bond_feats = [
                self._bond_types.index(bond.GetBondType()),
                self._bond_direction_types.index(bond.GetBondDir())
            ]
            edge_features.extend([bond_feats, bond_feats.copy()])

        if self._self_loop:
            self_loop_features = torch.zeros((mol.GetNumAtoms(), 2), dtype=torch.int64)
            self_loop_features[:, 0] = len(self._bond_types)

        if num_bonds == 0:
            edge_features = self_loop_features
        else:
            edge_features = np.stack(edge_features)
            edge_features = F.zerocopy_from_numpy(edge_features.astype(np.int64))
            if self._self_loop:
                edge_features = torch.cat([edge_features, self_loop_features], dim=0)

        return {'bond_type': edge_features[:, 0], 'bond_direction_type': edge_features[:, 1]}

class AttentiveFPBondFeaturizer(BaseBondFeaturizer):
    """The bond featurizer used in AttentiveFP

    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    The bond features include:
    * **One hot encoding of the bond type**. The supported bond types include
      ``SINGLE``, ``DOUBLE``, ``TRIPLE``, ``AROMATIC``.
    * **Whether the bond is conjugated.**.
    * **Whether the bond is in a ring of any size.**
    * **One hot encoding of the stereo configuration of a bond**. The supported bond stereo
      configurations include ``STEREONONE``, ``STEREOANY``, ``STEREOZ``, ``STEREOE``.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**

    Parameters
    ----------
    bond_data_field : str
        Name for storing bond features in DGLGraphs, default to ``'e'``.
    self_loop : bool
        Whether self loops will be added. Default to False. If True, it will use an additional
        column of binary values to indicate the identity of self loops. The feature of the
        self loops will be zero except for the additional column.

    Examples
    --------

    >>> from dgllife.utils import AttentiveFPBondFeaturizer
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='feat')
    >>> bond_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('feat')
    10

    >>> # Featurization with self loops to add
    >>> bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='feat', self_loop=True)
    >>> bond_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('feat')
    11

    See Also
    --------
    BaseBondFeaturizer
    CanonicalBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    PAGTNEdgeFeaturizer
    """
    def __init__(self, bond_data_field='e', self_loop=False):
        super(AttentiveFPBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [bond_type_one_hot,
                 bond_is_conjugated,
                 bond_is_in_ring,
                 partial(bond_stereo_one_hot, allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
                                                             Chem.rdchem.BondStereo.STEREOANY,
                                                             Chem.rdchem.BondStereo.STEREOZ,
                                                             Chem.rdchem.BondStereo.STEREOE])]
            )}, self_loop=self_loop)

class PAGTNEdgeFeaturizer(object):
    """The edge featurizer used in PAGTN

    PAGTN is introduced in
    `Path-Augmented Graph Transformer Network. <https://arxiv.org/abs/1905.12712>`__

    We build a complete graph and the edge features include:
    * **Shortest path between two nodes in terms of bonds**. To encode the path,
        we encode each bond on the path and concatenate their encodings. The encoding
        of a bond contains information about the bond type, whether the bond is
        conjugated and whether the bond is in a ring.
    * **One hot encoding of type of rings based on size and aromaticity**.
    * **One hot encoding of the distance between the nodes**.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_complete_graph` with
    self loops.**

    Parameters
    ----------
    bond_data_field : str
        Name for storing bond features in DGLGraphs, default to ``'e'``.
    max_length : int
        Maximum distance up to which shortest paths must be considered.
        Paths shorter than max_length will be padded and longer will be
        truncated, default to ``5``.

    Examples
    --------

    >>> from dgllife.utils import PAGTNEdgeFeaturizer
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = PAGTNEdgeFeaturizer(max_length=1)
    >>> bond_featurizer(mol)
    {'e': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size()
    14

    See Also
    --------
    BaseBondFeaturizer
    CanonicalBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    """
    def __init__(self, bond_data_field='e', max_length=5):
        self.bond_data_field = bond_data_field
        # Any two given nodes can belong to the same ring and here only
        # ring sizes of 5 and 6 are used. True & False indicate if it's aromatic or not.
        self.RING_TYPES = [(5, False), (5, True), (6, False), (6, True)]
        self.ordered_pair = lambda a, b: (a, b) if a < b else (b, a)
        self.bond_featurizer = ConcatFeaturizer([bond_type_one_hot,
                                                 bond_is_conjugated,
                                                 bond_is_in_ring])
        self.max_length = max_length

    def feat_size(self):
        """Get the feature size.

        Returns
        -------
        int
            Feature size.
        """
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self.bond_data_field]

        return feats.shape[-1]

    def bond_features(self, mol, path_atoms, ring_info):
        """Computes the edge features for a given pair of nodes.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.
        path_atoms: tuple
            Shortest path between the given pair of nodes.
        ring_info: list
            Different rings that contain the pair of atoms
        """
        features = []
        path_bonds = []
        path_length = len(path_atoms)
        for path_idx in range(path_length - 1):
            bond = mol.GetBondBetweenAtoms(path_atoms[path_idx], path_atoms[path_idx + 1])
            if bond is None:
                import warnings
                warnings.warn('Valid idx of bonds must be passed')
            path_bonds.append(bond)

        for path_idx in range(self.max_length):
            if path_idx < len(path_bonds):
                features.append(self.bond_featurizer(path_bonds[path_idx]))
            else:
                features.append([0, 0, 0, 0, 0, 0])

        if path_length + 1 > self.max_length:
            path_length = self.max_length + 1
        position_feature = np.zeros(self.max_length + 2)
        position_feature[path_length] = 1
        features.append(position_feature)
        if ring_info:
            rfeat = [one_hot_encoding(r, allowable_set=self.RING_TYPES) for r in ring_info]
            rfeat = [True] + np.any(rfeat, axis=0).tolist()
            features.append(rfeat)
        else:
            # This will return a boolean vector with all entries False
            features.append([False] + one_hot_encoding(ring_info, allowable_set=self.RING_TYPES))
        return np.concatenate(features, axis=0)

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping self._edge_data_field to a float32 tensor of shape (N, M), where
            N is the number of atom pairs and M is the feature size depending on max_length.
        """

        n_atoms = mol.GetNumAtoms()
        # To get the shortest paths between two nodes.
        paths_dict = {
            (i, j): Chem.rdmolops.GetShortestPath(mol, i, j)
            for i in range(n_atoms)
            for j in range(n_atoms)
            if i != j
            }
        # To get info if two nodes belong to the same ring.
        rings_dict = {}
        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        for ring in ssr:
            ring_sz = len(ring)
            is_aromatic = True
            for atom_idx in ring:
                if not mol.GetAtoms()[atom_idx].GetIsAromatic():
                    is_aromatic = False
                    break
            for ring_idx, atom_idx in enumerate(ring):
                for other_idx in ring[ring_idx:]:
                    atom_pair = self.ordered_pair(atom_idx, other_idx)
                    if atom_pair not in rings_dict:
                        rings_dict[atom_pair] = [(ring_sz, is_aromatic)]
                    else:
                        if (ring_sz, is_aromatic) not in rings_dict[atom_pair]:
                            rings_dict[atom_pair].append((ring_sz, is_aromatic))
        # Featurizer
        feats = []
        for i in range(n_atoms):
            for j in range(n_atoms):

                if (i, j) not in paths_dict:
                    feats.append(np.zeros(7*self.max_length + 7))
                    continue
                ring_info = rings_dict.get(self.ordered_pair(i, j), [])
                feats.append(self.bond_features(mol, paths_dict[(i, j)], ring_info))

        return {self.bond_data_field: torch.tensor(feats).float()}
