<%@ Page Title="" Language="vb" AutoEventWireup="false" MasterPageFile="~/Template2.Master" CodeBehind="Result2.aspx.vb" Inherits="CIS4290_Image_Classification_website.Result2" %>
<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
</asp:Content>
<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <div class="elementor-element elementor-element-8fbae7c elementor-column elementor-col-100 elementor-top-column" data-id="8fbae7c" data-element_type="column">
        <div class="elementor-column-wrap  elementor-element-populated">
            <div class="elementor-widget-wrap">
                <section class="elementor-element elementor-element-86056ae elementor-section-boxed elementor-section-height-default elementor-section-height-default elementor-section elementor-inner-section" data-id="86056ae" data-element_type="section">
                    <div class="elementor-container elementor-column-gap-default">
                        <div class="elementor-row">
                            <div class="elementor-element elementor-element-151f0c0 elementor-column elementor-col-100 elementor-inner-column" data-id="151f0c0" data-element_type="column">
                                <div class="elementor-column-wrap  elementor-element-populated">
                                    <div class="elementor-widget-wrap">
                                        <div class="elementor-element elementor-element-05b0559 elementor-widget elementor-widget-text-editor" data-id="05b0559" data-element_type="widget" data-widget_type="text-editor.default">
                                            <div class="elementor-widget-container">
                                                <div class="elementor-text-editor elementor-clearfix">
                                                    <h6>RESULTS OF IMAGE SET 2</h6>
                                                    <h2>This <span style="color: #f2ab41;">Image set</span> was much easier but had one problem.</h2>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="elementor-element elementor-element-ea49a3c elementor-widget elementor-widget-spacer" data-id="ea49a3c" data-element_type="widget" data-widget_type="spacer.default">
                                            <div class="elementor-widget-container">
                                                <div class="elementor-spacer">
                                                    <div class="elementor-spacer-inner"></div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
                <section class="elementor-element elementor-element-b5177b3 elementor-section-boxed elementor-section-height-default elementor-section-height-default elementor-section elementor-inner-section" data-id="b5177b3" data-element_type="section">
                    <div class="elementor-container elementor-column-gap-default">
                        <div class="elementor-row">
                            <div class="elementor-element elementor-element-d4b7954 elementor-column elementor-col-50 elementor-inner-column" data-id="d4b7954" data-element_type="column">
                                <div class="elementor-column-wrap  elementor-element-populated">
                                    <div class="elementor-widget-wrap">
                                        <div class="elementor-element elementor-element-bd83677 elementor-widget elementor-widget-text-editor" data-id="bd83677" data-element_type="widget" data-widget_type="text-editor.default">
                                            <div class="elementor-widget-container">
                                                <div class="elementor-text-editor elementor-clearfix">
                                                    <h4>This image set was really the first one we really tested our learning and understanding of Tensorflow.</h4>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="elementor-element elementor-element-478cf4b elementor-widget elementor-widget-text-editor" data-id="478cf4b" data-element_type="widget" data-widget_type="text-editor.default">
                                            <div class="elementor-widget-container">
                                                <div class="elementor-text-editor elementor-clearfix">
                                                    <p>
                                                        We got a high validation accuracy initially, but we still needed to learn how to adjust the model. 
                                                    </p>
                                                    <p>
                                                        All we did was adjust the image height and width, since it was technically not looking at the whole photo. We basically told the model to take in more data.
                                                    </p>
                                                    <p>
                                                        Doing so, allowed the model to reach 100% accuracy. We also believe this may also be attributed to the fact that this image set was from Kaggle.com.
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                                                                                
                                        <div class="elementor-element elementor-element-a7599d9 elementor-widget elementor-widget-spacer" data-id="a7599d9" data-element_type="widget" data-widget_type="spacer.default">
                                            <div class="elementor-widget-container">
                                                <div class="elementor-spacer">
                                                    <div class="elementor-spacer-inner"></div>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="elementor-element elementor-element-9580cdf elementor-widget__width-auto elementor-widget elementor-widget-button" data-id="9580cdf" data-element_type="widget" data-widget_type="button.default">
                                            <div class="elementor-widget-container">
                                                                                                        
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="elementor-element elementor-element-017b9c2 elementor-column elementor-col-50 elementor-inner-column" data-id="017b9c2" data-element_type="column">
                                <div class="elementor-column-wrap  elementor-element-populated">
                                    <div class="elementor-widget-wrap">
                                                                                                
                                        <div class="elementor-element elementor-element-80412c6 elementor-widget elementor-widget-image" data-id="80412c6" data-element_type="widget" data-widget_type="image.default">
                                            <div class="elementor-widget-container">
                                                <div class="elementor-image">
                                                    <img width="780" height="657" src="images/img2_ValidChart.jpg" class="attachment-large size-large" >
                                                </div>
                                            </div>
                                        </div>
                                                                                                
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
            </div>
        </div>
    </div>
</asp:Content>

<asp:Content ID="Content3" ContentPlaceHolderID="ContentPlaceHolder2" runat="server">
    <section>
            <div class="elementor-container elementor-column-gap-default">
                <div class="elementor-row">
                    
                    <img src="images/img2_Epoch.jpg"  />
                    <div class="elementor-element elementor-element-5e8b02b elementor-column elementor-col-50 elementor-inner-column" data-id="5e8b02b" data-element_type="column">
                        <div class="elementor-column-wrap">
                            <div class="elementor-widget-wrap"></div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
</asp:Content>

