package com.zlup.ide

import com.intellij.extapi.psi.PsiFileBase
import com.intellij.openapi.fileTypes.FileType
import com.intellij.psi.FileViewProvider

class ZlupFile(viewProvider: FileViewProvider) : PsiFileBase(viewProvider, ZlupLanguage) {
    override fun getFileType(): FileType = ZlupFileType
    override fun toString(): String = "Zlup File"
}
