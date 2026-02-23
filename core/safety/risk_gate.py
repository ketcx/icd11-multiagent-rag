"""Detects sensitive content and gracefully stops the session."""

import re

# Keywords triggering the RiskGate (Spanish and English)
RISK_KEYWORDS = [
    r"suicid",
    r"matarme",
    r"quitarme la vida",
    r"autolesion",
    r"hacerme daño",
    r"no quiero vivir",
    r"cortarme",
    r"self\.harm",
    r"kill myself",
]

# Educational system Spanish disclaimer string response
SAFE_RESPONSE_TEMPLATE = """
⚠️ NOTA DE SEGURIDAD: Se ha detectado contenido relacionado con
{risk_type}. Este es un sistema educativo y NO puede proporcionar
ayuda clínica real.

Si tú o alguien que conoces necesita ayuda inmediata:
- Línea de atención a la crisis: 024 (España)
- Teléfono de la Esperanza: 717 003 717
- Emergencias: 112

Esta sesión se ha pausado por seguridad.
"""


class RiskGate:
    """Intercepts sensitive content across all graph nodes."""

    def _classify_risk(self, pattern: str) -> str:
        # Default translation returned in Spanish for user conversational interface
        return "Riesgo de Autolesión o Suicidio"

    def check(self, text: str) -> tuple[bool, str | None]:
        """Returns (is_risky, risk_type) if sensitive content is detected."""
        for pattern in RISK_KEYWORDS:
            if re.search(pattern, text, re.IGNORECASE):
                return True, self._classify_risk(pattern)
        return False, None

    def get_safe_response(self, risk_type: str) -> str:
        """Returns a generic disclaimer and halts the response generation."""
        return SAFE_RESPONSE_TEMPLATE.format(risk_type=risk_type)
