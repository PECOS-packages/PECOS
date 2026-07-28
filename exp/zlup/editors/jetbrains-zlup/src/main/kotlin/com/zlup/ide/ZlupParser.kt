package com.zlup.ide

import com.intellij.lang.ASTNode
import com.intellij.lang.PsiBuilder
import com.intellij.lang.PsiParser
import com.intellij.psi.tree.IElementType

/**
 * Minimal parser for Zlup.
 * Actual parsing is handled by the LSP server - this just provides
 * basic token stream to PSI tree conversion.
 */
class ZlupParser : PsiParser {
    override fun parse(root: IElementType, builder: PsiBuilder): ASTNode {
        val rootMarker = builder.mark()

        while (!builder.eof()) {
            builder.advanceLexer()
        }

        rootMarker.done(root)
        return builder.treeBuilt
    }
}
