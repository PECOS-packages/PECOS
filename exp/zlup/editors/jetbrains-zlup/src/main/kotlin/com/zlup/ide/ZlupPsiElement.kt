package com.zlup.ide

import com.intellij.extapi.psi.ASTWrapperPsiElement
import com.intellij.lang.ASTNode

open class ZlupPsiElement(node: ASTNode) : ASTWrapperPsiElement(node)
