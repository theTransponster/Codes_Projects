import java.awt.Choice;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.ButtonGroup;
import javax.swing.ImageIcon;
import javax.swing.JButton;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JTextField;

import org.opencv.core.Core;

public class PanelPrincipal extends JPanel{

	public JTextField tNombre,
					   tEdad,
					   tTiempo,
					   tFecha;
	private JButton bIniciar;
	private Choice chPatologia;
	private VentanaPrincipal v;
	private JLabel lBienvenido,
				   lIngrese,
				   lNombre,
				   lEdad,
				   lTiempo,
				   lFecha,
				   lPatologia;
	private Image fondo;
	public int auxiliar;
	
	public PanelPrincipal(VentanaPrincipal v){
		super();
		this.setPreferredSize(new Dimension(600,400));
		this.v=v;
		this.setLayout(null);
		this.setBackground(Color.lightGray);
		this.fondo= new ImageIcon(getClass().getResource("angulobrazo.gif")).getImage();
		
		this.lBienvenido=new JLabel("Bienvenido");
		this.lBienvenido.setFont(new Font("TimesRoman", Font.BOLD,20));
		this.lBienvenido.setForeground(Color.white);
		this.add(this.lBienvenido);
		this.lBienvenido.setBounds(320, 35, 150, 15);
		
		this.lIngrese=new JLabel("Ingrese los datos del paciente ");
		this.lIngrese.setFont(new Font("TimesRoman", Font.BOLD,15));
		this.lIngrese.setForeground(Color.BLACK);
		this.add(this.lIngrese);
		this.lIngrese.setBounds(320, 80, 500, 20);
		
		this.lNombre=new JLabel("Nombre: ");
		this.lNombre.setFont(new Font("TimesRoman",Font.BOLD,15));
		this.lNombre.setForeground(Color.BLACK);
		this.add(this.lNombre);
		this.lNombre.setBounds(320,115,100,20);
		this.tNombre=new JTextField(100);
		this.add(this.tNombre);
		this.tNombre.setBounds(420, 115, 100, 20);
		
		this.lEdad=new JLabel("Edad: ");
		this.lEdad.setFont(new Font("TimesRoman",Font.BOLD,15));
		this.lEdad.setForeground(Color.BLACK);
		this.add(this.lEdad);
		this.lEdad.setBounds(320,150,100,20);
		this.tEdad=new JTextField(100);
		this.add(this.tEdad);
		this.tEdad.setBounds(420,150,100,20);
		
		this.lTiempo=new JLabel("Terapia num: ");
		this.lTiempo.setFont(new Font("TimesRoman",Font.BOLD,15));
		this.lTiempo.setForeground(Color.BLACK);
		this.add(this.lTiempo);
		this.lTiempo.setBounds(320, 185, 100, 20);
		this.tTiempo=new JTextField(30);
		this.add(this.tTiempo);
		this.tTiempo.setBounds(420, 185, 100, 20);
		
		this.lFecha=new JLabel("Fecha: ");
		this.lFecha.setFont(new Font("TimesRoman",Font.BOLD,15));
		this.lFecha.setForeground(Color.BLACK);
		this.add(this.lFecha);
		this.lFecha.setBounds(320, 220, 100, 20);
		this.tFecha=new JTextField(100);
		this.add(this.tFecha);
		this.tFecha.setBounds(420, 220, 100, 20);
		
		this.lPatologia=new JLabel("Patología a tratar: ");
		this.lPatologia.setFont(new Font("TimesRoman",Font.BOLD,15));
		this.lPatologia.setForeground(Color.BLACK);
		this.add(this.lPatologia);
		this.lPatologia.setBounds(320, 255, 200, 20);
		this.chPatologia=new Choice();
		this.chPatologia.setSize(50,20);
		this.chPatologia.add("Sobre-Flexión");
		this.chPatologia.add("Sobre-Extensión");
		this.add(this.chPatologia);
		this.chPatologia.setBounds(320, 290, 200, 20);
		this.bIniciar=new JButton("Iniciar Terapia");
		this.bIniciar.setBounds(450, 340, 120, 30);
		this.bIniciar.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				
				if(chPatologia.getSelectedItem()=="Sobre-Flexión"){
					auxiliar=1;
				}else{
					auxiliar=2;
				}
				VentanaHipoFlexion v1=new VentanaHipoFlexion();
				v1.setDatos(PanelPrincipal.this);
				PanelPrincipal.this.v.dispose();
			}
		}); this.add(this.bIniciar);
	}

	public void paintComponent(Graphics g){
		super.paintComponent(g);
		g.drawImage(this.fondo,0,0,300,400,this);
	}
	
}
