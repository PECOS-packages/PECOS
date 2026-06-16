package com.zlup.ide

import com.intellij.lang.Language

object ZlupLanguage : Language("Zlup") {
    override fun getDisplayName(): String = "Zlup"
    override fun isCaseSensitive(): Boolean = true
}
