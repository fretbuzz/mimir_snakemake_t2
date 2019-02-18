# doesn't do much ATM, but in like 2 hours I could automate the whole setup for wordpress...

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
import sys
import time
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver import FirefoxProfile
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import unittest, time, re
import os,errno
from multiprocessing.dummy import Pool as ThreadPool
import threading
from selenium.webdriver.firefox.options import Options
import shutil

def install_pluggin():
    try:
        faker_press_install = driver.find_element_by_id('plugin_install_from_iframe')
        print "faker_press_install", faker_press_install
        # faker_press_install = driver.find_element_by_class_name('install-now button')
        # for elem in login_form:
        #    print 'element.text: {0}'.format(elem)
        print faker_press_install
        faker_press_install.click()

        activate_button()

    except Exception as e:
        print("there was an exception thrown in installing/activating pluggin...", e)
        pass  # might already be installed...

def activate_button():
    class_of_activate_button = "button-primary"
    faker_press_activate = driver.find_element_by_class_name(class_of_activate_button)
    print "faker_press_activate", faker_press_activate
    faker_press_activate.click()

def terms_page_code():
    driver.find_element_by_id("fakerpress-field-qty-min").click()
    driver.find_element_by_id("fakerpress-field-qty-min").clear()
    driver.find_element_by_id("fakerpress-field-qty-min").send_keys("300")
    driver.find_element_by_id("fakerpress-field-size-min").click()
    driver.find_element_by_id("fakerpress-field-size-min").clear()
    driver.find_element_by_id("fakerpress-field-size-min").send_keys("3")
    try:
        driver.find_element_by_xpath("//div[@id='s2id_fakerpress-field-taxonomies']/ul").click()
        #driver.find_element_by_id("s2id_autogen1").click()
        driver.find_element_by_id('select2-result-label-3').click()
    except:
        pass
    try:
        driver.find_element_by_xpath("//div[@id='s2id_fakerpress-field-taxonomies']/ul").click()
        driver.find_element_by_id('select2-result-label-10').click()
    except:
        pass
    try:
        driver.find_element_by_xpath("//div[@id='s2id_fakerpress-field-taxonomies']/ul").click()
        driver.find_element_by_id('select2-result-label-17').click()
    except:
        pass
    #driver.find_element_by_xpath("//div[@id='s2id_fakerpress-field-taxonomies']/ul").click()
    #driver.find_element_by_xpath("//input[@value='Generate']").click()
    activate_button()

def posts_page_code():
    driver.find_element_by_xpath("(//a[contains(text(),'Posts')])[2]").click()
    driver.find_element_by_id("fakerpress-field-qty-min").click()
    driver.find_element_by_id("fakerpress-field-qty-min").clear()
    driver.find_element_by_id("fakerpress-field-qty-min").send_keys("800")

    driver.find_element_by_xpath("//tr[@id='fakerpress-field-post_types-container']/td").click()
    driver.find_element_by_xpath("//div[@id='s2id_fakerpress-field-post_types']/ul").click()
    driver.find_element_by_id('select2-result-label-14').click()

    meta_type_menu_id = 's2id_fakerpress-field-meta-type'
    driver.find_element_by_id(meta_type_menu_id).click()
    driver.find_element_by_id('select2-result-label-21').click()
    activate_button()

def comments_page_code():
    #driver.find_element_by_link_text("Comments").click()
    driver.find_element_by_id("fakerpress-field-qty-min").click()
    driver.find_element_by_id("fakerpress-field-qty-min").clear()
    driver.find_element_by_id("fakerpress-field-qty-min").send_keys("1600")
    driver.find_element_by_xpath("//div[@id='s2id_fakerpress-field-post_types']/ul").click()
    driver.find_element_by_id("select2-result-label-7").click()
    #driver.find_element_by_xpath("//input[@value='Generate']").click()
    activate_button()

def download_csv(url_to_download_csv):
    driver.get(url_to_download_csv)

def export_urls_code(driver_val):
    #driver.find_element_by_xpath("//li[@id='menu-plugins']/a/div[3]").click()
    driver_val.find_element_by_link_text("Export All URLs").click()
    driver_val.find_element_by_name("post-type").click()
    driver_val.find_element_by_name("additional-data[]").click()
    driver_val.find_element_by_name("export-type").click()
    driver_val.find_element_by_name("export").click()
    driver_val.find_element_by_xpath("//form[@id='infoForm']/table/tbody/tr[3]/td").click()
    #link_to_get_csv = driver.find_element_by_link_text("Click here") #find_element_by_class_name("updated")
    time.sleep(25)
    link_to_get_csv = driver_val.find_element_by_link_text("Click here")
    print "link_to_get_csv",link_to_get_csv, link_to_get_csv.get_attribute('href')
    url_to_download_csv = link_to_get_csv.get_attribute('href')
    url_to_download_csv = url_to_download_csv.replace('http', 'https')
    print "new link", url_to_download_csv
    #link_to_get_csv.click()
    print "about to download csv...", url_to_download_csv

    #driver.get(url_to_download_csv)
    #pool = ThreadPool(1)
    #thread = threading.Thread(target = download_csv, args=(url_to_download_csv,))
    #results = pool.map(download_csv, url_to_download_csv)
    #driver.navigate().to(url_to_download_csv)
    #thread.start()


    time.sleep(10)
    driver_val.get(url_to_download_csv)
    print "just downloaded csv...."

def make_new_application_passwd(driver_val):
    '''
    driver.find_element_by_xpath("//li[@id='menu-users']/a/div[3]").click()
    driver.find_element_by_id("user-search-input").click()
    driver.find_element_by_id("user-search-input").clear()
    driver.find_element_by_id("user-search-input").send_keys("user")
    driver.find_element_by_id("user-search-input").send_keys(Keys.ENTER)

    ## TODO: this is very annoying, but it must be done...
    driver.find_element_by_link_text("user").click()
    '''
    user_profile_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/profile.php?wp_http_referer=%2Fwp-admin%2Fusers.php%3Fs%3Duser%26action%3D-1%26new_role%26paged%3D1%26action2%3D-1%26new_role2'
    driver_val.get(user_profile_page)

    #driver.find_element_by_xpath("//a[contains(text(),'user')]").click()
    driver_val.find_element_by_name("new_application_password_name").click()
    driver_val.find_element_by_name("new_application_password_name").clear()
    driver_val.find_element_by_name("new_application_password_name").send_keys("wp_test")
    driver_val.find_element_by_id("do_new_application_password").click()
    #driver.find_element_by_xpath("//div[@id='application-passwords-section']/div[2]/div/div/div/kbd").click()
    #driver.find_element_by_xpath("//div[@id='application-passwords-section']/div[2]/div/div/div").click()
    #driver.find_element_by_xpath("//div[@id='application-passwords-section']/div[2]/div/div/button").click()
    #pwd = driver.find_element_by_class_name('app-pass-dialog notification-dialog')
    time.sleep(2)
    pwd = driver_val.find_element_by_class_name("new-application-password-content")
    pwd = pwd.text.split(":")[-1].rstrip().lstrip()
    print "pwd", pwd #get_attribute('body')
                                      #new-application-password-content
    return pwd

def user_page_code():
    roles_field_class = 'select2-input'#''select2-choices'
    rolesBox = driver.find_element_by_class_name(roles_field_class)
    rolesBox.send_keys('Administrator')
    rolesBox.send_keys(Keys.RETURN)
    rolesBox.send_keys('Editor')
    rolesBox.send_keys(Keys.RETURN)
    rolesBox.send_keys('Author')
    rolesBox.send_keys(Keys.RETURN)
    rolesBox.send_keys('Contributor')
    rolesBox.send_keys(Keys.RETURN)
    rolesBox.send_keys('Subscriber')
    rolesBox.send_keys(Keys.RETURN)
    max_quantity_box = driver.find_element_by_id(min_num)
    print "min_num_box: ",min_num
    max_quantity_box.send_keys("400")
    activate_button()

def admin_login(admin_pwd, driver_val):
    driver_val.get('https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin')
    #assert "Python" in driver.title
    #elem = driver.find_element_by_name("q")
    elem = driver_val.find_element_by_name("log")
    elem.clear()
    elem.send_keys("user")
    elem = driver_val.find_element_by_name("pwd")
    elem.clear()
    elem.send_keys(admin_pwd)
    elem.send_keys(Keys.RETURN)
    print elem

def main(ip_of_wp, port_of_wp, admin_pwd):
    global driver
    global driver_two

    try:
        os.makedirs('./wp_csv_loc')
    except OSError as e:
        print e
    # if the dictory already exists,then we want to clear it (to make the result easy to find)
	# (taken from: https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder-in-python)
	#file_name = None
    for file_name in os.listdir('./wp_csv_loc'):
        file_path = os.path.join('./wp_csv_loc', file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
        if e.errno != errno.EEXIST:
            raise

    options = Options()
    options.headless = True

    # from: https://selenium-python.readthedocs.io/faq.html (literally copy-pasted)
    fp = webdriver.FirefoxProfile()
    fp.set_preference("browser.download.folderList", 2)
    fp.set_preference("browser.download.manager.showWhenStarting", False)
    fp.set_preference("browser.download.dir", os.getcwd() + '/wp_csv_loc/')
    fp.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/octet-stream")
    fp.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")

    driver = webdriver.Firefox(fp, options=options)
    driver_two = webdriver.Firefox(fp, options=options)
    admin_login(admin_pwd, driver)
    admin_login(admin_pwd, driver_two)
    time.sleep(5)
    

    # okay, first install fakerpress
    #'''
    page_about_fakerpress = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/plugin-install.php?tab=plugin-information&plugin=fakerpress&TB_iframe=true&height=-34%22&width=772'
    #print driver.page_source.encode("utf-8")
    driver.get(page_about_fakerpress)
    time.sleep(5)
    install_pluggin()
    time.sleep(10)

    # then install "Export All URLs"
    page_about_export_all_urls = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/plugin-install.php?tab=plugin-information&plugin=export-all-urls&TB_iframe=true&width=772&height=627'
    driver.get(page_about_export_all_urls)
    install_pluggin()
    time.sleep(10)

    # then install "app_pass"
    page_about_app_pass = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/plugin-install.php?tab=plugin-information&plugin=application-passwords&TB_iframe=true&width=772&height=627'
    driver.get(page_about_app_pass)
    install_pluggin()
    #'''

    # now generate the fake data using fakerpress
    max_num = 'fakerpress-field-qty-max'
    max_num_class = 'fp-field fp-type-number fp-size-tiny'
    global min_num 
    min_num = 'fakerpress-field-qty-min'
    drop_down_id = 's2id_fakerpress-field-meta-type'

    #''' # this is good.
    user_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/admin.php?page=fakerpress&view=users'
    driver.get(user_page)
    user_page_code()
    time.sleep(170)
    '''

    #''' # this is good.
    terms_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/admin.php?page=fakerpress&view=terms'
    driver.get(terms_page)
    terms_page_code()
    time.sleep(60)
    '''
    #''' # this is good
    post_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/admin.php?page=fakerpress&view=posts'
    driver.get(post_page)
    posts_page_code()
    time.sleep(300)
    #'''
    #''' # this is good
    comments_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/admin.php?page=fakerpress&view=comments'
    driver.get(comments_page)
    comments_page_code()
    time.sleep(300)
    #'''

    export_all_urls_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/options-general.php?page=extract-all-urls-settings'
    #''' This is good
    driver.get(export_all_urls_page)
    #export_urls_code(driver_two)
    thread = threading.Thread(target = export_urls_code, args=(driver,))
    thread.start()

    print "will now sleep for 15"
    time.sleep(15)
    print "done sleeping"
    #'''
    new_pdw = make_new_application_passwd(driver_two)
    time.sleep(35)

    # todo: first finish loading wp (via fakerpress) -- okay, I think this might be done (just gotta test it...)
    # then modify so that passwd is a cmd line arg -- okay, I think this might be done (just gotta test it...)
    # then make it run on cloudlab <----- start from here
    # then modify so called before run_experiment (should be easy...)
    # and needs to modify wordpress_background to take the passwd from this function as a cmdline argument...
        # might need to write to a file or something... (b/c gets wierd with python scripting)
            # might need to get tricky but shouldn't be too bad either...

    driver.close()
    driver_two.close()

    # let's also return the name of the resulting csv folder...
    folders_in_csv_path = os.listdir('./wp_csv_loc')
    print "folders_in_csv_path", folders_in_csv_path
    path_to_csv_file = './wp_csv_loc/' + folders_in_csv_path[0]
    shutil.copy(path_to_csv_file, "../" + "wordpress_users.csv")
    shutil.copy(path_to_csv_file, "../load_generators/" + "wordpress_users.csv")
    
    with open('../load_generators/wordpress_api_pwd.txt', 'w') as f:
        f.write(new_pdw)

    return new_pdw#, path_to_csv_file

if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print "needs an ip address, budy"

    print sys.argv
    ip_of_wp = sys.argv[1]
    port_of_wp = sys.argv[2]
    admin_pwd = sys.argv[3]

    main(ip_of_wp, port_of_wp, admin_pwd)
