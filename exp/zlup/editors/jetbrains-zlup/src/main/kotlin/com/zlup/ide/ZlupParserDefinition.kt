package com.zlup.ide

import com.intellij.lang.ASTNode
import com.intellij.lang.ParserDefinition
import com.intellij.lang.PsiParser
import com.intellij.lexer.Lexer
import com.intellij.openapi.project.Project
import com.intellij.psi.FileViewProvider
import com.intellij.psi.PsiElement
import com.intellij.psi.PsiFile
import com.intellij.psi.tree.IFileElementType
import com.intellij.psi.tree.TokenSet

class ZlupParserDefinition : ParserDefinition {
    companion object {
        val FILE = IFileElementType(ZlupLanguage)
    }

    override fun createLexer(project: Project): Lexer = ZlupLexer()

    override fun getCommentTokens(): TokenSet = ZlupTokenTypes.COMMENTS

    override fun getStringLiteralElements(): TokenSet = ZlupTokenTypes.STRINGS

    override fun createParser(project: Project): PsiParser {
        // We use LSP for parsing, so we provide a minimal parser
        return ZlupParser()
    }

    override fun getFileNodeType(): IFileElementType = FILE

    override fun createFile(viewProvider: FileViewProvider): PsiFile = ZlupFile(viewProvider)

    override fun createElement(node: ASTNode): PsiElement = ZlupPsiElement(node)
}
