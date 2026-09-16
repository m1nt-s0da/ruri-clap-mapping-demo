from slopmachine.embedding.ruri_v3_30m_int8 import RuriV3_30M_Int8
from slopmachine.embedding.ruri_v3_130m_int8 import RuriV3_130M_Int8
from slopmachine.embedding.laion_clap_text_int8 import LaionClapTextInt8

_clap_text_encoder_instance: LaionClapTextInt8 | None = None
_ruriv3_encoder_instance: RuriV3_30M_Int8 | None = None
_ruriv3_encoder_130m_instance: RuriV3_130M_Int8 | None = None


def clap_text_encoder():
    global _clap_text_encoder_instance
    if _clap_text_encoder_instance is None:
        _clap_text_encoder_instance = LaionClapTextInt8()
    return _clap_text_encoder_instance


def ruriv3_encoder():
    global _ruriv3_encoder_instance
    if _ruriv3_encoder_instance is None:
        _ruriv3_encoder_instance = RuriV3_30M_Int8()
    return _ruriv3_encoder_instance


def ruriv3_encoder_130m():
    global _ruriv3_encoder_130m_instance
    if _ruriv3_encoder_130m_instance is None:
        _ruriv3_encoder_130m_instance = RuriV3_130M_Int8()
    return _ruriv3_encoder_130m_instance
