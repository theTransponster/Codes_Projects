import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JColorChooser;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JSlider;
import javax.swing.JTextField;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

public class ControlesPocoyo extends JPanel {

	private PanelDibujo mipanel;
	private JButton saludar,
					cambiar,
					selarchivo,
					cambiarglobo,
					cambiarnave,
					guardar;
	private JTextField nombre;
	private JRadioButton azul,
						 verde,
						 rojo;
	private JSlider lista;
	private JFileChooser fc,
						 fs;
	
	
	public ControlesPocoyo(PanelDibujo mipanel){
		super();
		this.mipanel=mipanel;
		this.setPreferredSize(new Dimension(150,600));
		
		this.nombre=new JTextField(12);
		this.add(this.nombre);
		
		this.saludar=new JButton("Saludar");
		this.saludar.addActionListener(new ActionListener(){
				public void actionPerformed(ActionEvent e){
					String nombre=ControlesPocoyo.this.nombre.getText();
					mipanel.nombre(nombre);
					JOptionPane.showMessageDialog(null, "Hola"+nombre);
				}
		});
		this.add(this.saludar);
		
		this.azul=new JRadioButton("Color Azul");
		this.add(this.azul);
		
		this.verde=new JRadioButton("Color verde");
		this.add(this.verde);
		
		this.rojo=new JRadioButton("Color rojo");
		this.add(this.rojo);
		
		ButtonGroup bg=new ButtonGroup();
		bg.add(this.verde);
		bg.add(this.rojo);
		bg.add(this.azul);
		
		this.cambiar=new JButton("Cambiar fondo");
		this.cambiar.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				if(rojo.isSelected()){
					setBackground(Color.red);
				}
				if(verde.isSelected()){
					setBackground(Color.green);
				}
				if(azul.isSelected()){
					setBackground(Color.BLUE);		
				}
			}
		});this.add(this.cambiar);
		
		this.cambiarglobo=new JButton("Cambiar globo");
		this.cambiarglobo.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				Color color=JColorChooser.showDialog(mipanel, "Selecciona el color del globo de la nave", mipanel.getColor());
				mipanel.setColor(color);
			
			}
		});this.add(this.cambiarglobo);
		
		this.cambiarnave=new JButton("Cambiar nave");
		this.cambiarnave.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				Color colornave=JColorChooser.showDialog(mipanel, "Seleccione color de nave", mipanel.getColornave());
				mipanel.setColornave(colornave);
			}
		});this.add(this.cambiarnave);
		
		
		this.selarchivo=new JButton("Seleccionar archivo");
		this.selarchivo.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				fc.showOpenDialog(null);
				File o=fc.getSelectedFile();
				JOptionPane.showMessageDialog(null, "Archivo seleccionado: "+o);
			}
		});
		this.add(this.selarchivo);
		this.fc=new JFileChooser();
		
		this.guardar=new JButton("Guardar imagen");
		this.guardar.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				fs.showSaveDialog(null);
				File s=fs.getSelectedFile();
				Dimension size=mipanel.getSize();
				BufferedImage bi=new BufferedImage(size.width,size.height,BufferedImage.TYPE_INT_RGB);
				Graphics2D g2=bi.createGraphics();
				mipanel.paint(g2);
				try {
					ImageIO.write(bi, "png", s);
				} catch (IOException e1) {
					// TODO Auto-generated catch block
					e1.getMessage();
				}
			}
		}); this.add(this.guardar);
		this.fs=new JFileChooser();
		
		this.lista=new JSlider(JSlider.VERTICAL,0,150,0);
		
		this.lista.setMajorTickSpacing(15);
		this.lista.setMinorTickSpacing(5);
		this.lista.setPaintLabels(true);
		this.lista.setPaintTicks(true);
		this.lista.addChangeListener(new ChangeListener() {
			
			@Override
			public void stateChanged(ChangeEvent e) {
				//System.out.println(slider.getValue());
				mipanel.setiVY(lista.getValue());
				
			}
		});
		this.add(this.lista);
	}
	
	
}
