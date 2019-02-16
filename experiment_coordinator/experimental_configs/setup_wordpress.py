# doesn't do much ATM, but in like 2 hours I could automate the whole setup for wordpress...

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
import sys
import time
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import unittest, time, re

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
    except:
        print("there was an exception thrown in installing/activating pluggin...")
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

    ## TODO: Meta Field Rules needs to be modified so taht the attachemnt doesn't do stuff anymore...

    #driver.find_element_by_xpath("//input[@value='Generate']").click()
    activate_button()
def comments_page_code():
    driver.find_element_by_link_text("Comments").click()
    driver.find_element_by_id("fakerpress-field-qty-min").click()
    driver.find_element_by_id("fakerpress-field-qty-min").clear()
    driver.find_element_by_id("fakerpress-field-qty-min").send_keys("1600")
    driver.find_element_by_xpath("//div[@id='s2id_fakerpress-field-post_types']/ul").click()
    #driver.find_element_by_xpath("//input[@value='Generate']").click()
    activate_button()
def export_urls_code():
    #driver.find_element_by_xpath("//li[@id='menu-plugins']/a/div[3]").click()
    driver.find_element_by_link_text("Export All URLs").click()
    driver.find_element_by_name("post-type").click()
    driver.find_element_by_name("additional-data[]").click()
    driver.find_element_by_name("export-type").click()
    driver.find_element_by_name("export").click()
    driver.find_element_by_xpath("//form[@id='infoForm']/table/tbody/tr[3]/td").click()
    #link_to_get_csv = driver.find_element_by_link_text("Click here") #find_element_by_class_name("updated")
    link_to_get_csv = driver.find_element_by_link_text("Click here")
    print "link_to_get_csv",link_to_get_csv, link_to_get_csv.get_attribute('href')
    url_to_download_csv = link_to_get_csv.get_attribute('href')
    url_to_download_csv = url_to_download_csv.replace('http', 'https')
    print "new link", url_to_download_csv
    #link_to_get_csv.click()
    driver.get(url_to_download_csv)

def make_new_application_passwd():
    '''
    driver.find_element_by_xpath("//li[@id='menu-users']/a/div[3]").click()
    driver.find_element_by_id("user-search-input").click()
    driver.find_element_by_id("user-search-input").clear()
    driver.find_element_by_id("user-search-input").send_keys("user")
    driver.find_element_by_id("user-search-input").send_keys(Keys.ENTER)

    ## TODO: this is very annoying, but it must be done...
    driver.find_element_by_link_text("user").click()
    '''
    user_profile_page = 'https://192.168.64.4:31423/wp-admin/profile.php?wp_http_referer=%2Fwp-admin%2Fusers.php%3Fs%3Duser%26action%3D-1%26new_role%26paged%3D1%26action2%3D-1%26new_role2'
    driver.get(user_profile_page)

    #driver.find_element_by_xpath("//a[contains(text(),'user')]").click()
    driver.find_element_by_name("new_application_password_name").click()
    driver.find_element_by_name("new_application_password_name").clear()
    driver.find_element_by_name("new_application_password_name").send_keys("wp_test")
    driver.find_element_by_id("do_new_application_password").click()
    #driver.find_element_by_xpath("//div[@id='application-passwords-section']/div[2]/div/div/div/kbd").click()
    #driver.find_element_by_xpath("//div[@id='application-passwords-section']/div[2]/div/div/div").click()
    #driver.find_element_by_xpath("//div[@id='application-passwords-section']/div[2]/div/div/button").click()
    #pwd = driver.find_element_by_class_name('app-pass-dialog notification-dialog')
    time.sleep(2)
    pwd = driver.find_element_by_class_name("new-application-password-content")
    pwd = pwd.text.split(":")[-1].rstrip().lstrip()
    print "pwd", pwd #get_attribute('body')
                                      #new-application-password-content
    return pwd

if __name__== "__main__":
    if len(sys.argv) <= 2:
        print "needs an ip address, budy"

    print sys.argv
    ip_of_wp = sys.argv[1]
    port_of_wp = sys.argv[2]

    #firefox_binary=FirefoxBinary('/anaconda2/lib/python2.7/site-packages/selenium/webdriver/firefox')
    driver = webdriver.Firefox()
    #driver.get("http://www.python.org")
    driver.get('https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin')
    #assert "Python" in driver.title
    #elem = driver.find_element_by_name("q")
    elem = driver.find_element_by_name("log")
    elem.clear()
    elem.send_keys("user")
    elem = driver.find_element_by_name("pwd")
    elem.clear()
    elem.send_keys("HSjDUn1Vlt")
    elem.send_keys(Keys.RETURN)
    print elem
    time.sleep(5)
    #driver.get('https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/plugin-install.php?tab=search&s=faker')

    # okay, first install fakerpress
    '''
    page_about_fakerpress = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/plugin-install.php?tab=plugin-information&plugin=fakerpress&TB_iframe=true&height=-34%22&width=772'
    driver.get(page_about_fakerpress)
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
    '''

    # now generate the fake data using fakerpress
    max_num = 'fakerpress-field-qty-max'
    max_num_class = 'fp-field fp-type-number fp-size-tiny'
    min_num = 'fakerpress-field-qty-min'
    drop_down_id = 's2id_fakerpress-field-meta-type'

    '''
    user_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/admin.php?page=fakerpress&view=users'
    driver.get(user_page)
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
    time.sleep(170)
    '''

    '''
    terms_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/admin.php?page=fakerpress&view=terms'
    driver.get(terms_page)
    taxonamy_selector_id = 's2id_fakerpress-field-taxonomies'
    tax_class = 'select2-choices'
    taxonamyBox = driver.find_element_by_class_name(tax_class)
    taxonamyBox.click()
    #tax_class_two = 'select2-drop-mask'
    #taxonamyBox_two = driver.find_element_by_class_name(tax_class_two)
    #taxonamyBox_two.click()
    #taxonamyBox_two.click()
    #taxonamyBox_two.click()
    time.sleep(300)
    taxonamyBox.send_keys('Categories')
    taxonamyBox.send_keys(Keys.RETURN)
    taxonamyBox.send_keys('Tags')
    taxonamyBox.send_keys(Keys.RETURN)
    taxonamyBox.send_keys('Format')
    taxonamyBox.send_keys(Keys.RETURN)
    max_quantity_box = driver.find_element_by_id(min_num)
    max_quantity_box.send_keys("300")
    term_size_box = driver.find_element_by_id('fakerpress-field-size-min')
    max_quantity_box.send_keys("3")
    activate_button()
    time.sleep(120)

    post_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/admin.php?page=fakerpress&view=posts'
    driver.get(post_page)
    post_types_id = 's2id_fakerpress-field-post_types'
    postTypeBox = driver.find_element_by_id(post_types_id)
    postTypeBox.send_keys('Pages')
    postTypeBox.send_keys(Keys.RETURN)
    max_quantity_box = driver.find_element_by_id(min_num)
    print "min_num_box: ", min_num
    max_quantity_box.send_keys("8000")
    activate_button()
    time.sleep(500)

    comments_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/admin.php?page=fakerpress&view=comments'
    driver.get(comments_page)
    # post type add pages
    post_types_id = 's2id_fakerpress-field-post_types'
    postTypeBox = driver.find_element_by_id(post_types_id)
    postTypeBox.send_keys('Pages')
    postTypeBox.send_keys(Keys.RETURN)
    max_quantity_box = driver.find_element_by_id(min_num)
    print "min_num_box: ", min_num
    max_quantity_box.send_keys("1600")
    activate_button()
    time.sleep(300)
    '''
    terms_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/admin.php?page=fakerpress&view=terms'
    driver.get(terms_page)
    #terms_page_code()
    #time.sleep(60)
    '''
    post_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/admin.php?page=fakerpress&view=posts'
    driver.get(post_page)
    posts_page_code()
    time.sleep(300)
    comments_page_code()
    time.sleep(300)
    '''
    export_all_urls_page = 'https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/options-general.php?page=extract-all-urls-settings'
    #driver.get(export_all_urls_page)
    #export_urls_code()
    #time.sleep(15)
    new_pdw = make_new_application_passwd()
    time.sleep(45)

    # todo: 3 more pages of fakerpress nonsense... followed by exporting the urls
    # and then creating an app password...
    # modify woordpress_background to take the password
    # and then moving exporting urls and moving them to the right spot.
    # and then wrapping this whole thing into a system that can be deployed on cloudlab easily...

    # todo: first finish loading wp (via fakerpress)
    # then modify so that passwd is a cmd line arg
    # then modify so called before run_experiment
    # and needs to modify wordpress_background to take the passwd from this function as a cmdline argument...

    driver.close()
    return new_pwd