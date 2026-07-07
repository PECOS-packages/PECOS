package com.zlup.ide

import com.intellij.lexer.Lexer
import com.intellij.openapi.editor.DefaultLanguageHighlighterColors
import com.intellij.openapi.editor.HighlighterColors
import com.intellij.openapi.editor.colors.TextAttributesKey
import com.intellij.openapi.editor.colors.TextAttributesKey.createTextAttributesKey
import com.intellij.openapi.fileTypes.SyntaxHighlighter
import com.intellij.openapi.fileTypes.SyntaxHighlighterBase
import com.intellij.openapi.fileTypes.SyntaxHighlighterFactory
import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.psi.tree.IElementType

class ZlupSyntaxHighlighter : SyntaxHighlighterBase() {
    companion object {
        val KEYWORD = createTextAttributesKey("ZLUP_KEYWORD", DefaultLanguageHighlighterColors.KEYWORD)
        val TYPE = createTextAttributesKey("ZLUP_TYPE", DefaultLanguageHighlighterColors.CLASS_NAME)
        val GATE = createTextAttributesKey("ZLUP_GATE", DefaultLanguageHighlighterColors.FUNCTION_CALL)
        val IDENTIFIER = createTextAttributesKey("ZLUP_IDENTIFIER", DefaultLanguageHighlighterColors.IDENTIFIER)
        val NUMBER = createTextAttributesKey("ZLUP_NUMBER", DefaultLanguageHighlighterColors.NUMBER)
        val STRING = createTextAttributesKey("ZLUP_STRING", DefaultLanguageHighlighterColors.STRING)
        val LINE_COMMENT = createTextAttributesKey("ZLUP_LINE_COMMENT", DefaultLanguageHighlighterColors.LINE_COMMENT)
        val BLOCK_COMMENT = createTextAttributesKey("ZLUP_BLOCK_COMMENT", DefaultLanguageHighlighterColors.BLOCK_COMMENT)
        val OPERATOR = createTextAttributesKey("ZLUP_OPERATOR", DefaultLanguageHighlighterColors.OPERATION_SIGN)
        val BRACKETS = createTextAttributesKey("ZLUP_BRACKETS", DefaultLanguageHighlighterColors.BRACKETS)
        val BRACES = createTextAttributesKey("ZLUP_BRACES", DefaultLanguageHighlighterColors.BRACES)
        val PARENTHESES = createTextAttributesKey("ZLUP_PARENTHESES", DefaultLanguageHighlighterColors.PARENTHESES)
        val COMMA = createTextAttributesKey("ZLUP_COMMA", DefaultLanguageHighlighterColors.COMMA)
        val SEMICOLON = createTextAttributesKey("ZLUP_SEMICOLON", DefaultLanguageHighlighterColors.SEMICOLON)
        val DOT = createTextAttributesKey("ZLUP_DOT", DefaultLanguageHighlighterColors.DOT)
        val BAD_CHARACTER = createTextAttributesKey("ZLUP_BAD_CHARACTER", HighlighterColors.BAD_CHARACTER)

        private val KEYWORD_KEYS = arrayOf(KEYWORD)
        private val TYPE_KEYS = arrayOf(TYPE)
        private val GATE_KEYS = arrayOf(GATE)
        private val IDENTIFIER_KEYS = arrayOf(IDENTIFIER)
        private val NUMBER_KEYS = arrayOf(NUMBER)
        private val STRING_KEYS = arrayOf(STRING)
        private val COMMENT_KEYS = arrayOf(LINE_COMMENT)
        private val BLOCK_COMMENT_KEYS = arrayOf(BLOCK_COMMENT)
        private val OPERATOR_KEYS = arrayOf(OPERATOR)
        private val BRACKET_KEYS = arrayOf(BRACKETS)
        private val BRACE_KEYS = arrayOf(BRACES)
        private val PAREN_KEYS = arrayOf(PARENTHESES)
        private val COMMA_KEYS = arrayOf(COMMA)
        private val SEMICOLON_KEYS = arrayOf(SEMICOLON)
        private val DOT_KEYS = arrayOf(DOT)
        private val BAD_CHAR_KEYS = arrayOf(BAD_CHARACTER)
        private val EMPTY_KEYS = emptyArray<TextAttributesKey>()
    }

    override fun getHighlightingLexer(): Lexer = ZlupLexer()

    override fun getTokenHighlights(tokenType: IElementType): Array<TextAttributesKey> {
        return when (tokenType) {
            ZlupTokenTypes.KEYWORD -> KEYWORD_KEYS
            ZlupTokenTypes.TYPE -> TYPE_KEYS
            ZlupTokenTypes.GATE -> GATE_KEYS
            ZlupTokenTypes.IDENTIFIER -> IDENTIFIER_KEYS
            ZlupTokenTypes.NUMBER -> NUMBER_KEYS
            ZlupTokenTypes.STRING -> STRING_KEYS
            ZlupTokenTypes.LINE_COMMENT -> COMMENT_KEYS
            ZlupTokenTypes.BLOCK_COMMENT -> BLOCK_COMMENT_KEYS
            ZlupTokenTypes.OPERATOR, ZlupTokenTypes.ARROW -> OPERATOR_KEYS
            ZlupTokenTypes.LBRACKET, ZlupTokenTypes.RBRACKET -> BRACKET_KEYS
            ZlupTokenTypes.LBRACE, ZlupTokenTypes.RBRACE -> BRACE_KEYS
            ZlupTokenTypes.LPAREN, ZlupTokenTypes.RPAREN -> PAREN_KEYS
            ZlupTokenTypes.COMMA -> COMMA_KEYS
            ZlupTokenTypes.SEMICOLON -> SEMICOLON_KEYS
            ZlupTokenTypes.DOT -> DOT_KEYS
            ZlupTokenTypes.BAD_CHARACTER -> BAD_CHAR_KEYS
            else -> EMPTY_KEYS
        }
    }
}

class ZlupSyntaxHighlighterFactory : SyntaxHighlighterFactory() {
    override fun getSyntaxHighlighter(project: Project?, virtualFile: VirtualFile?): SyntaxHighlighter {
        return ZlupSyntaxHighlighter()
    }
}
