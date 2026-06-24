package com.zlup.ide

import com.intellij.psi.tree.IElementType
import com.intellij.psi.tree.TokenSet

object ZlupTokenTypes {
    // Keywords
    @JvmField val KEYWORD = ZlupTokenType("KEYWORD")

    // Identifiers
    @JvmField val IDENTIFIER = ZlupTokenType("IDENTIFIER")

    // Literals
    @JvmField val NUMBER = ZlupTokenType("NUMBER")
    @JvmField val STRING = ZlupTokenType("STRING")

    // Comments
    @JvmField val LINE_COMMENT = ZlupTokenType("LINE_COMMENT")
    @JvmField val BLOCK_COMMENT = ZlupTokenType("BLOCK_COMMENT")

    // Operators
    @JvmField val OPERATOR = ZlupTokenType("OPERATOR")

    // Brackets
    @JvmField val LPAREN = ZlupTokenType("LPAREN")
    @JvmField val RPAREN = ZlupTokenType("RPAREN")
    @JvmField val LBRACE = ZlupTokenType("LBRACE")
    @JvmField val RBRACE = ZlupTokenType("RBRACE")
    @JvmField val LBRACKET = ZlupTokenType("LBRACKET")
    @JvmField val RBRACKET = ZlupTokenType("RBRACKET")

    // Other
    @JvmField val DOT = ZlupTokenType("DOT")
    @JvmField val COMMA = ZlupTokenType("COMMA")
    @JvmField val SEMICOLON = ZlupTokenType("SEMICOLON")
    @JvmField val COLON = ZlupTokenType("COLON")
    @JvmField val ARROW = ZlupTokenType("ARROW")
    @JvmField val AT = ZlupTokenType("AT")

    // Whitespace and bad characters
    @JvmField val WHITE_SPACE = ZlupTokenType("WHITE_SPACE")
    @JvmField val BAD_CHARACTER = ZlupTokenType("BAD_CHARACTER")

    // Types (built-in)
    @JvmField val TYPE = ZlupTokenType("TYPE")

    // Quantum gates
    @JvmField val GATE = ZlupTokenType("GATE")

    // Token sets
    @JvmField val COMMENTS = TokenSet.create(LINE_COMMENT, BLOCK_COMMENT)
    @JvmField val STRINGS = TokenSet.create(STRING)
    @JvmField val WHITESPACES = TokenSet.create(WHITE_SPACE)
}

class ZlupTokenType(debugName: String) : IElementType(debugName, ZlupLanguage)
