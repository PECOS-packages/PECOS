package com.zlup.ide

import com.intellij.openapi.fileTypes.LanguageFileType
import javax.swing.Icon

object ZlupFileType : LanguageFileType(ZlupLanguage) {
    override fun getName(): String = "Zlup"
    override fun getDescription(): String = "Zlup quantum programming language"
    override fun getDefaultExtension(): String = "zlp"
    override fun getIcon(): Icon = ZlupIcons.FILE

    const val EXTENSION = "zlp"
}
