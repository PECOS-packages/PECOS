package com.zlup.ide

import com.intellij.codeInsight.template.TemplateActionContext
import com.intellij.codeInsight.template.TemplateContextType

class ZlupTemplateContextType : TemplateContextType("Zlup") {
    override fun isInContext(templateActionContext: TemplateActionContext): Boolean {
        val file = templateActionContext.file
        return file.name.endsWith(".zlp")
    }
}
