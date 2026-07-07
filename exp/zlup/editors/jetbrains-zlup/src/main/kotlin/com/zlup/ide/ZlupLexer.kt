package com.zlup.ide

import com.intellij.lexer.LexerBase
import com.intellij.psi.tree.IElementType

class ZlupLexer : LexerBase() {
    private var buffer: CharSequence = ""
    private var startOffset: Int = 0
    private var endOffset: Int = 0
    private var currentOffset: Int = 0
    private var tokenStart: Int = 0
    private var tokenEnd: Int = 0
    private var tokenType: IElementType? = null

    companion object {
        private val KEYWORDS = setOf(
            // Control flow
            "fn", "if", "else", "while", "for", "return", "break", "continue",
            // Declarations
            "const", "var", "pub", "struct", "enum", "union", "error", "type",
            // Error handling
            "try", "catch", "orelse", "defer", "errdefer",
            // Modifiers
            "mut", "comptime", "inline", "packed", "extern",
            // Logical operators
            "and", "or", "not",
            // Literals
            "true", "false", "null", "undefined", "unit",
            // Other
            "in", "tick", "barrier", "fault"
        )

        private val TYPES = setOf(
            // Integer types
            "u1", "u2", "u4", "u8", "u16", "u32", "u64", "u128",
            "i8", "i16", "i32", "i64", "i128",
            "usize", "isize",
            // Float types
            "f16", "f32", "f64", "f128",
            // Angle type
            "a64",
            // Boolean
            "bool",
            // Quantum types
            "qubit", "bit", "qalloc",
            // Special
            "void", "type", "anytype", "anyerror", "anyfault"
        )

        private val GATES = setOf(
            // Single-qubit gates (lowercase)
            "h", "x", "y", "z", "s", "sdg", "t", "tdg",
            "sx", "sxdg", "sy", "sydg", "sz", "szdg",
            "f", "fdg", "f4", "f4dg",
            // Rotations
            "rx", "ry", "rz",
            // Two-qubit gates
            "cx", "cy", "cz", "ch", "swap", "iswap",
            "sxx", "sxxdg", "syy", "syydg", "szz", "szzdg", "rzz",
            // Three-qubit gates
            "ccx",
            // Measurement/preparation
            "mz", "mx", "my", "pz", "px", "py"
        )
    }

    override fun start(buffer: CharSequence, startOffset: Int, endOffset: Int, initialState: Int) {
        this.buffer = buffer
        this.startOffset = startOffset
        this.endOffset = endOffset
        this.currentOffset = startOffset
        advance()
    }

    override fun getState(): Int = 0

    override fun getTokenType(): IElementType? = tokenType

    override fun getTokenStart(): Int = tokenStart

    override fun getTokenEnd(): Int = tokenEnd

    override fun advance() {
        tokenStart = currentOffset

        if (currentOffset >= endOffset) {
            tokenType = null
            tokenEnd = currentOffset
            return
        }

        val c = buffer[currentOffset]

        when {
            // Whitespace
            c.isWhitespace() -> {
                while (currentOffset < endOffset && buffer[currentOffset].isWhitespace()) {
                    currentOffset++
                }
                tokenType = ZlupTokenTypes.WHITE_SPACE
            }

            // Line comment
            c == '/' && currentOffset + 1 < endOffset && buffer[currentOffset + 1] == '/' -> {
                currentOffset += 2
                while (currentOffset < endOffset && buffer[currentOffset] != '\n') {
                    currentOffset++
                }
                tokenType = ZlupTokenTypes.LINE_COMMENT
            }

            // Block comment
            c == '/' && currentOffset + 1 < endOffset && buffer[currentOffset + 1] == '*' -> {
                currentOffset += 2
                while (currentOffset + 1 < endOffset) {
                    if (buffer[currentOffset] == '*' && buffer[currentOffset + 1] == '/') {
                        currentOffset += 2
                        break
                    }
                    currentOffset++
                }
                tokenType = ZlupTokenTypes.BLOCK_COMMENT
            }

            // String
            c == '"' -> {
                currentOffset++
                while (currentOffset < endOffset) {
                    val ch = buffer[currentOffset]
                    if (ch == '"') {
                        currentOffset++
                        break
                    }
                    if (ch == '\\' && currentOffset + 1 < endOffset) {
                        currentOffset += 2
                    } else {
                        currentOffset++
                    }
                }
                tokenType = ZlupTokenTypes.STRING
            }

            // Character literal
            c == '\'' -> {
                currentOffset++
                while (currentOffset < endOffset) {
                    val ch = buffer[currentOffset]
                    if (ch == '\'') {
                        currentOffset++
                        break
                    }
                    if (ch == '\\' && currentOffset + 1 < endOffset) {
                        currentOffset += 2
                    } else {
                        currentOffset++
                    }
                }
                tokenType = ZlupTokenTypes.STRING
            }

            // Number
            c.isDigit() -> {
                while (currentOffset < endOffset) {
                    val ch = buffer[currentOffset]
                    if (ch.isLetterOrDigit() || ch == '.' || ch == '_') {
                        currentOffset++
                    } else {
                        break
                    }
                }
                tokenType = ZlupTokenTypes.NUMBER
            }

            // Identifier or keyword
            c.isLetter() || c == '_' -> {
                while (currentOffset < endOffset) {
                    val ch = buffer[currentOffset]
                    if (ch.isLetterOrDigit() || ch == '_') {
                        currentOffset++
                    } else {
                        break
                    }
                }
                val word = buffer.subSequence(tokenStart, currentOffset).toString()
                tokenType = when {
                    word in KEYWORDS -> ZlupTokenTypes.KEYWORD
                    word in TYPES -> ZlupTokenTypes.TYPE
                    word in GATES -> ZlupTokenTypes.GATE
                    else -> ZlupTokenTypes.IDENTIFIER
                }
            }

            // Arrow
            c == '-' && currentOffset + 1 < endOffset && buffer[currentOffset + 1] == '>' -> {
                currentOffset += 2
                tokenType = ZlupTokenTypes.ARROW
            }

            // Brackets and punctuation
            c == '(' -> { currentOffset++; tokenType = ZlupTokenTypes.LPAREN }
            c == ')' -> { currentOffset++; tokenType = ZlupTokenTypes.RPAREN }
            c == '{' -> { currentOffset++; tokenType = ZlupTokenTypes.LBRACE }
            c == '}' -> { currentOffset++; tokenType = ZlupTokenTypes.RBRACE }
            c == '[' -> { currentOffset++; tokenType = ZlupTokenTypes.LBRACKET }
            c == ']' -> { currentOffset++; tokenType = ZlupTokenTypes.RBRACKET }
            c == '.' -> { currentOffset++; tokenType = ZlupTokenTypes.DOT }
            c == ',' -> { currentOffset++; tokenType = ZlupTokenTypes.COMMA }
            c == ';' -> { currentOffset++; tokenType = ZlupTokenTypes.SEMICOLON }
            c == ':' -> { currentOffset++; tokenType = ZlupTokenTypes.COLON }
            c == '@' -> { currentOffset++; tokenType = ZlupTokenTypes.AT }

            // Operators
            c in "+-*/%=!<>&|^~" -> {
                while (currentOffset < endOffset && buffer[currentOffset] in "+-*/%=!<>&|^~") {
                    currentOffset++
                }
                tokenType = ZlupTokenTypes.OPERATOR
            }

            // Bad character
            else -> {
                currentOffset++
                tokenType = ZlupTokenTypes.BAD_CHARACTER
            }
        }

        tokenEnd = currentOffset
    }

    override fun getBufferSequence(): CharSequence = buffer

    override fun getBufferEnd(): Int = endOffset
}
