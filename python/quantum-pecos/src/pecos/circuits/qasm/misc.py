# Copyright 2021 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


class Comment:
    """Represents a comment."""

    def __init__(self, *texts) -> None:
        if isinstance(texts, str):
            self.texts = [texts]
        else:
            self.texts = list(texts)

    def __str__(self) -> str:
        # TODO: Just insert a new blank line every time a <br> is seen. e.g., e

        lines = []
        for t in self.texts:
            for ln in t.split("\n"):
                ln = ln.strip()

                if ln == "":
                    lines.append("//")
                else:
                    ln = ln.replace("<br>", "</><br></>").split("</>")

                    for ln2 in ln:
                        ln2 = ln2.strip()

                        if ln2 == "":
                            pass
                        elif ln2 == "<br>":
                            lines.append("")
                        else:
                            lines.append(f"// {ln2}")

        return "\n".join(lines)


br = ""
