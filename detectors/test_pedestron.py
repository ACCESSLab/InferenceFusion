import subprocess
import shlex

def run_command(command):
    print("[+] starting subprocess ", command)
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, bufsize=1)
    while True:
        output = process.stdout.readline().decode("utf-8")
        if output == '' and process.poll() is not None:
            break
        if output:
            print ("[+ Pedestron]: ", output.strip())
        if "terminated" in output:
            break
    rc = process.poll()
    print("[-] finished subprocess ", command)
    return rc

if __name__ == '__main__':
    run_command("sh runObjectDetection.sh")
