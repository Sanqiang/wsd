"""
Helper functions for MIMIC-III dataset.

"""
from preprocess.text_helper import sub_patterns


# DeID replacement for MIMIC (ShARe/CLEF)
def sub_deid_patterns(txt):
    # DATE
    txt = sub_patterns(txt, [
        # normal date
        r"\[\*\*(\d{4}-)?\d{1,2}-\d{1,2}\*\*\]",
        # date range
        r"\[\*\*Date [rR]ange.+\*\*\]",
        # month/year
        r"\[\*\*-?\d{1,2}-?/\d{4}\*\*\]",
        # year
        r"\[\*\*(Year \([24] digits\).+)\*\*\]",
        # holiday
        r"\[\*\*Holiday.+\*\*\]",
        # XXX-XX-XX
        r"\[\*\*\d{3}-\d{1,2}-\d{1,2}\*\*\]",
        # date with format
        r"\[\*\*(Month(/Day)?(/Year)?|Year(/Month)?(/Day)?|Day Month).+\*\*\]",
        # uppercase month year
        r"\[\*\*(January|February|March|April|May|June|July|August|September|October|November|December).+\*\*\]",
    ], "DATE-DEID")

    # NAME
    txt = sub_patterns(txt, [
        # name
        r"\[\*\*(First |Last )?Name.+\*\*\]",
        # name initials
        r"\[\*\*Initial.+\*\*\]",
        # name with sex
        r"\[\*\*(Female|Male).+\*\*\]",
        # doctor name
        r"\[\*\*Doctor.+\*\*\]",
        # known name
        r"\[\*\*Known.+\*\*\]",
        # wardname
        r"\[\*\*Wardname.+\*\*\]",
    ], "NAME-DEID")

    # INSTITUTION
    txt = sub_patterns(txt, [
        # hospital
        r"\[\*\*Hospital.+\*\*\]",
        # university
        r"\[\*\*University.+\*\*\]",
        # company
        r"\[\*\*Company.+\*\*\]",
    ], "INSTITUTION-DEID")

    # clip number
    txt = sub_patterns(txt, [
        r"\[\*\*Clip Number.+\*\*\]",
    ], "CLIP-NUMBER-DEID")

    # digits
    txt = sub_patterns(txt, [
        r"\[\*\* ?\d{1,5}\*\*\]",
    ], "DIGITS-DEID")

    # tel/fax
    txt = sub_patterns(txt, [
        r"\[\*\*Telephone/Fax.+\*\*\]",
        r"\[\*\*\*\*\]",
    ], "PHONE-DEID")

    # EMPTY
    txt = sub_patterns(txt, [
        r"\[\*\* ?\*\*\]",
    ], "EMPTY-DEID")

    # numeric identifier
    txt = sub_patterns(txt, [
        r"\[\*\*Numeric Identifier.+\*\*\]",
    ], "NUMERIC-DEID")

    # AGE
    txt = sub_patterns(txt, [
        r"\[\*\*Age.+\*\*\]",
    ], "AGE-DEID")

    # PLACE
    txt = sub_patterns(txt, [
        # country
        r"\[\*\*Country.+\*\*\]",
        # state
        r"\[\*\*State.+\*\*\]",
    ], "PLACE-DEID")

    # STREET-ADDRESS
    txt = sub_patterns(txt, [
        r"\[\*\*Location.+\*\*\]",
        r"\[\*\*.+ Address.+\*\*\]",
    ], "STREET-ADDRESS-DEID")

    # MD number
    txt = sub_patterns(txt, [
        r"\[\*\*MD Number.+\*\*\]",
    ], "MD-NUMBER-DEID")

    # other numbers
    txt = sub_patterns(txt, [
        # job
        r"\[\*\*Job Number.+\*\*\]",
        # medical record number
        r"\[\*\*Medical Record Number.+\*\*\]",
        # SSN
        r"\[\*\*Social Security Number.+\*\*\]",
        # unit number
        r"\[\*\*Unit Number.+\*\*\]",
        # pager number
        r"\[\*\*Pager number.+\*\*\]",
        # serial number
        r"\[\*\*Serial Number.+\*\*\]",
        # provider number
        r"\[\*\*Provider Number.+\*\*\]",
    ], "OTHER-NUMBER-DEID")

    # info
    txt = sub_patterns(txt, [
        r"\[\*\*.+Info.+\*\*\]",
    ], "INFO-DEID")

    # E-mail
    txt = sub_patterns(txt, [
        r"\[\*\*E-mail address.+\*\*\]",
        r"\[\*\*URL.+\*\*\]"
    ], "EMAIL-DEID")

    # other
    txt = sub_patterns(txt, [
        r"\[\*\*(.*)?\*\*\]",
    ], "OTHER-DEID")
    return txt
