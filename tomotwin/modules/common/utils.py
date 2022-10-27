def check_for_updates():
    """
    Checks if cryolo updates are available
    :return: None
    """
    try:
        import tomotwin
        import urllib.request
        import json

        with urllib.request.urlopen(
            url="https://pypi.org/pypi/tomotwin-cryoet/json", timeout=5
        ) as url:
            data = json.loads(url.read().decode())
            from packaging import version

            current_version = tomotwin.__version__
            latest_version = current_version
            for ver in data["releases"].keys():
                vers = version.parse(ver)
                if vers > version.parse(latest_version) and vers.is_prerelease == False:
                    latest_version = ver
            if version.parse(latest_version) > version.parse(current_version):
                print("###############################################")
                print("New version of TomoTwin available")
                print("Local version:\t\t", current_version)
                print("Latest version:\t\t", latest_version)
                print(
                    "More information here:\n",
                    "https://tomotwin-cryoet.readthedocs.io/en/stable/changes.html",
                )
                print("###############################################")
            else:

                if version.parse(current_version).is_devrelease or version.parse(current_version).is_prerelease:
                    print("Uuuuh! You are brave and running a development version of TomoTwin. "+u'\u2665 \u2665'+" Thanks for testing" +u'\u2665 \u2665')
                else:
                    print("Latest version of TomoTwin is installed :-)")
    except Exception as e:
        print("###############################################")
        print("Skip version check, as it failed with the following error:")
        print(e)
        print("###############################################")
        pass
