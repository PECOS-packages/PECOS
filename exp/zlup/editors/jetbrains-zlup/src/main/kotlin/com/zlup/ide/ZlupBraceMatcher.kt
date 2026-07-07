package com.zlup.ide

import com.intellij.lang.BracePair
import com.intellij.lang.PairedBraceMatcher
import com.intellij.psi.PsiFile
import com.intellij.psi.tree.IElementType

class ZlupBraceMatcher : PairedBraceMatcher {
    companion object {
        private val PAIRS = arrayOf(
            BracePair(ZlupTokenTypes.LBRACE, ZlupTokenTypes.RBRACE, true),
            BracePair(ZlupTokenTypes.LPAREN, ZlupTokenTypes.RPAREN, false),
            BracePair(ZlupTokenTypes.LBRACKET, ZlupTokenTypes.RBRACKET, false)
        )
    }

    override fun getPairs(): Array<BracePair> = PAIRS

    override fun isPairedBracesAllowedBeforeType(lbraceType: IElementType, contextType: IElementType?): Boolean = true

    override fun getCodeConstructStart(file: PsiFile, openingBraceOffset: Int): Int = openingBraceOffset
}
